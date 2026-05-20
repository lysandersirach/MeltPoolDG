#include <meltpooldg/phase_change/evaporation_mass_flux_operator_thickness_integration.hpp>
//
#include <deal.II/base/array_view.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/mpi_remote_point_evaluation.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/types.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/vector_operation.h>

#include <deal.II/numerics/vector_tools_evaluate.h>

#include <meltpooldg/level_set/level_set_tools.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>

#include <fstream>
#include <numeric>
#include <string>
#include <vector>


namespace MeltPoolDG::Evaporation
{
  using namespace dealii;

  static int count = 0;

  template <int dim, typename number>
  EvaporationMassFluxOperatorThicknessIntegration<dim, number>::
    EvaporationMassFluxOperatorThicknessIntegration(
      const ScratchData<dim, dim, number>          &scratch_data,
      const EvaporationModelBase<number>           &evaporation_model,
      const EvaporationThicknessIntegrationData    &thickness_integration_data,
      const LevelSet::ReinitializationData<number> &reinit_data,
      const VectorType                             &level_set_as_heaviside,
      const BlockVectorType                        &normal_vector,
      const unsigned int                            ls_hanging_nodes_dof_idx,
      const unsigned int                            normal_dof_idx,
      const unsigned int                            heat_hanging_nodes_dof_idx,
      const unsigned int                            evapor_mass_flux_dof_idx)
    : scratch_data(scratch_data)
    , evaporation_model(evaporation_model)
    , thickness_integration_data(thickness_integration_data)
    , reinit_data(reinit_data)
    , level_set_as_heaviside(level_set_as_heaviside)
    , normal_vector(normal_vector)
    , ls_hanging_nodes_dof_idx(ls_hanging_nodes_dof_idx)
    , normal_dof_idx(normal_dof_idx)
    , heat_hanging_nodes_dof_idx(heat_hanging_nodes_dof_idx)
    , evapor_mass_flux_dof_idx(evapor_mass_flux_dof_idx)
    , fe_dim(FE_Q<dim>(scratch_data.get_degree(normal_dof_idx)), dim)
  {}

  template <int dim, typename number>
  void
  EvaporationMassFluxOperatorThicknessIntegration<dim, number>::compute_evaporative_mass_flux(
    VectorType       &evaporative_mass_flux,
    const VectorType &temperature) const
  {
    Assert(dim > 1, ExcMessage("The requested operation can only be performed for dim>1."));

    if constexpr (dim > 1)
      {
        scratch_data.initialize_dof_vector(evaporative_mass_flux, evapor_mass_flux_dof_idx);
        /*
         * generate point cloud normal to interface
         */
        std::vector<Point<dim>>   global_points_normal_to_interface;
        std::vector<unsigned int> global_points_normal_to_interface_pointer;

        const auto thickness_integration_band =
          5 * reinit_data.compute_interface_thickness_parameter_epsilon(
                scratch_data.get_min_cell_size());

        Assert(thickness_integration_band > 0.0,
               ExcMessage("Thickness for interface integration not set."));

        LevelSet::Tools::generate_points_along_normal<dim>(
          global_points_normal_to_interface,
          global_points_normal_to_interface_pointer,
          scratch_data.get_dof_handler(ls_hanging_nodes_dof_idx),
          fe_dim,
          scratch_data.get_mapping(),
          level_set_as_heaviside,
          normal_vector,
          thickness_integration_band,
          thickness_integration_data.subdivisions_per_side,
          /* bidirectional */ true,
          /* contour_value */ 0.5,
          thickness_integration_data.subdivisions_MCA);

        const bool update_ghosts = !level_set_as_heaviside.has_ghost_elements();
        if (update_ghosts)
          level_set_as_heaviside.update_ghost_values();
        const bool heat_update_ghosts = !temperature.has_ghost_elements();
        if (heat_update_ghosts)
          temperature.update_ghost_values();


        if (false)
          {
            /*
             * debug
             */
            const auto global_points_normal_to_interface_all =
              Utilities::MPI::reduce<std::vector<Point<dim>>>(global_points_normal_to_interface,
                                                              scratch_data.get_mpi_comm(),
                                                              [](const auto &a, const auto &b) {
                                                                auto result = a;
                                                                result.insert(result.end(),
                                                                              b.begin(),
                                                                              b.end());
                                                                return result;
                                                              });

            std::ofstream myfile;
            myfile.open(std::to_string(count) + "_generated_points.dat");

            for (const auto &p : global_points_normal_to_interface_all)
              {
                for (unsigned int d = 0; d < dim; ++d)
                  myfile << p[d] << " ";
                myfile << std::endl;
              }

            myfile.close();
            count++;
          }
        /*
         * evaluate result at points normal to interface
         */
        const typename Utilities::MPI::RemotePointEvaluation<dim, dim>::AdditionalData
          additional_data(1e-6 /*tolerance*/, true /*unique mapping*/);

        Utilities::MPI::RemotePointEvaluation<dim, dim> remote_point_evaluation(additional_data);

        remote_point_evaluation.reinit(global_points_normal_to_interface,
                                       scratch_data.get_triangulation(),
                                       scratch_data.get_mapping());


        const auto temperature_evaluation_values =
          dealii::VectorTools::point_values<1>(remote_point_evaluation,
                                               scratch_data.get_dof_handler(
                                                 heat_hanging_nodes_dof_idx),
                                               temperature);

        const std::vector<Tensor<1, dim, number>> level_set_gradient_values =
          dealii::VectorTools::point_gradients<1>(remote_point_evaluation,
                                                  scratch_data.get_dof_handler(
                                                    ls_hanging_nodes_dof_idx),
                                                  level_set_as_heaviside);
        /*
         * evaluate line integral
         *
         * @todo: how to determine start and end point for a given level set
         * value better than 5*epsilon?
         *
         */
        std::vector<number> mass_flux_val;
        mass_flux_val.resize(global_points_normal_to_interface_pointer.back());

        // loop over points located at the interface and determined by means of MCA
        for (unsigned int i = 0; i < global_points_normal_to_interface_pointer.size() - 1; ++i)
          {
            const auto start = global_points_normal_to_interface_pointer[i];
            const auto end   = global_points_normal_to_interface_pointer[i + 1];
            const auto size  = end - start;

            std::vector<number> func_vals(size);

            // loop over all points along normal at one MC point
            for (unsigned int l = 0; l < size; ++l)
              {
                func_vals[l] = evaporation_model.local_compute_evaporative_mass_flux(
                                 temperature_evaluation_values[start + l]) *
                               level_set_gradient_values[start + l].norm();
              }

            number mass_flux_average = UtilityFunctions::integrate_over_line<dim>(
              func_vals,
              std::vector(global_points_normal_to_interface.begin() + start,
                          global_points_normal_to_interface.begin() + end));

            /*
             * set for each point along the interface the value equal to the one determined
             * by the line integral
             */
            for (unsigned int l = 0; l < size; ++l)
              mass_flux_val[start + l] = mass_flux_average;
          }

        /*
         * Broadcast values determined along the normal direction to nodal points by means
         * of a simple averaging procedure. Subsequently, a DoF-vector is filled.
         *
         * @note We also tried the extrapolation via a radial basis function, however it was
         * too sensitive with respect to the inherent distance parameter.
         */
        VectorType vector_multiplicity;
        vector_multiplicity.reinit(evaporative_mass_flux);

        const auto broadcast_function = [&](const auto &values, const auto &cell_data) {
          // loop over all cells where points along normal are interior
          for (unsigned int i = 0; i < cell_data.cells.size(); ++i)
            {
              typename DoFHandler<dim>::active_cell_iterator cell = {
                &remote_point_evaluation.get_triangulation(),
                cell_data.cells[i].first,
                cell_data.cells[i].second,
                &scratch_data.get_dof_handler(ls_hanging_nodes_dof_idx)};

              // values at the points along normal in reference coordinates
              const ArrayView<const number> heat_vals(values.data() +
                                                        cell_data.reference_point_ptrs[i],
                                                      cell_data.reference_point_ptrs[i + 1] -
                                                        cell_data.reference_point_ptrs[i]);

              // extract dof indices of cell
              std::vector<types::global_dof_index> local_dof_indices(
                cell->get_fe().n_dofs_per_cell());
              cell->get_dof_indices(local_dof_indices);

              /*
               * Compute mean value from the values of points within the cell and set the values at
               * the cell nodes equal to the latter.
               */
              Vector<number> nodal_values(cell->get_fe().n_dofs_per_cell());

              const number average =
                std::accumulate(heat_vals.begin(), heat_vals.end(), 0.0) / heat_vals.size();

              for (auto &n : nodal_values)
                n = average;

              scratch_data.get_constraint(evapor_mass_flux_dof_idx)
                .distribute_local_to_global(nodal_values, local_dof_indices, evaporative_mass_flux);

              // count the number of written values (multiplicity)
              for (auto &val : nodal_values)
                val = 1.0;

              scratch_data.get_constraint(ls_hanging_nodes_dof_idx)
                .distribute_local_to_global(nodal_values, local_dof_indices, vector_multiplicity);
            }
        };

        // fill dof vector
        std::vector<number> buffer;

        remote_point_evaluation.template process_and_evaluate<number>(mass_flux_val,
                                                                      buffer,
                                                                      broadcast_function);

        // take care dofs shared by multiple geometric entities
        evaporative_mass_flux.compress(VectorOperation::add);
        vector_multiplicity.compress(VectorOperation::add);

        for (unsigned int i = 0; i < vector_multiplicity.locally_owned_size(); ++i)
          if (vector_multiplicity.local_element(i) > 1.0)
            evaporative_mass_flux.local_element(i) /= vector_multiplicity.local_element(i);

        scratch_data.get_constraint(evapor_mass_flux_dof_idx).distribute(evaporative_mass_flux);

        if (heat_update_ghosts)
          temperature.zero_out_ghost_values();

        if (update_ghosts)
          level_set_as_heaviside.zero_out_ghost_values();
      }
  }

  template class EvaporationMassFluxOperatorThicknessIntegration<1, double>;
  template class EvaporationMassFluxOperatorThicknessIntegration<2, double>;
  template class EvaporationMassFluxOperatorThicknessIntegration<3, double>;
} // namespace MeltPoolDG::Evaporation
