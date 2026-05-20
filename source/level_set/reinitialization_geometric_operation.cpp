#include <deal.II/base/exceptions.h>
#include <deal.II/base/timer.h>

#include <deal.II/numerics/vector_tools_interpolate.h>

#include <meltpooldg/level_set/reinitialization_geometric_operation.hpp>
#include <meltpooldg/utilities/characteristic_functions.hpp>
#include <meltpooldg/utilities/iteration_monitor.hpp>
#include <meltpooldg/utilities/journal.hpp>
#include <meltpooldg/utilities/scoped_name.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

namespace MeltPoolDG::LevelSet
{
  using namespace dealii;

  template <int dim, typename number>
  ReinitializationGeometricOperation<dim, number>::ReinitializationGeometricOperation(
    const ScratchData<dim, dim, number> &scratch_data_in,
    const ReinitializationData<number>  &reinit_data_in,
    const unsigned int                   ls_dof_idx_in,
    const unsigned int                   ls_quad_idx_in)
    : scratch_data(scratch_data_in)
    , reinit_data(reinit_data_in)
    , ls_dof_idx(ls_dof_idx_in)
    , ls_quad_idx(ls_quad_idx_in)
  {}

  template <int dim, typename number>
  void
  ReinitializationGeometricOperation<dim, number>::create_solver()
  {
    auto dof_handler_ptr = std::shared_ptr<DoFHandler<dim>>(
      const_cast<DoFHandler<dim> *>(&scratch_data.get_dof_handler(ls_dof_idx)), [](auto *) {});

    // We assume that the input level set is centered around zero
    const double iso_level = 0.0;

    signed_distance_solver =
      std::make_unique<SignedDistanceSolver<dim>>(dof_handler_ptr,
                                                  reinit_data.geometric.max_distance,
                                                  iso_level,
                                                  1.0 /*scaling to the level set function*/,
                                                  reinit_data.geometric.verbosity == 0 ?
                                                    Verbosity::quiet :
                                                    Verbosity::verbose);
  }

  template <int dim, typename number>
  void
  ReinitializationGeometricOperation<dim, number>::reinit()
  {
    if (not signed_distance_solver)
      create_solver();

    scratch_data.initialize_dof_vector(level_set, ls_dof_idx);
    signed_distance_solver->setup_dofs();
  }

  template <int dim, typename number>
  void
  ReinitializationGeometricOperation<dim, number>::set_initial_condition(
    const VectorType &level_set_in)
  {
    level_set.copy_locally_owned_data_from(level_set_in);
    level_set.update_ghost_values();
  }

  template <int dim, typename number>
  void
  ReinitializationGeometricOperation<dim, number>::set_initial_condition(
    const Function<dim> &initial_field_function)
  {
    level_set.zero_out_ghost_values();

    dealii::VectorTools::interpolate(scratch_data.get_mapping(),
                                     scratch_data.get_dof_handler(ls_dof_idx),
                                     initial_field_function,
                                     level_set);

    scratch_data.get_constraint(ls_dof_idx).distribute(level_set);
    level_set.update_ghost_values();
  }

  template <int dim, typename number>
  void
  ReinitializationGeometricOperation<dim, number>::solve()
  {
    const ScopedName         scope_n("solve");
    const TimerOutput::Scope scope_t(scratch_data.get_timer(), scope_n);

    signed_distance_solver->set_level_set_from_background_mesh(
      scratch_data.get_dof_handler(ls_dof_idx), level_set);

    signed_distance_solver->solve();

    // transform to tanh function
    if (reinit_data.geometric.transform_to_tanh)
      transform_signed_distance_to_tanh_level_set();

    Journal::print_formatted_norm<number>(
      scratch_data.get_pcout(1),
      [&]() -> number {
        return VectorTools::compute_norm<dim, number>(get_level_set(),
                                                      scratch_data,
                                                      ls_dof_idx,
                                                      ls_quad_idx);
      },
      "level set",
      "reinitialization_geometric",
      15);

    Journal::print_line(scratch_data.get_pcout(2),
                        "Geometric signed-distance reinitialization completed.",
                        "reinitialization_geometric");
  }

  template <int dim, typename number>
  void
  ReinitializationGeometricOperation<dim, number>::transform_signed_distance_to_tanh_level_set()
  {
    scratch_data.initialize_dof_vector(level_set, ls_dof_idx);

    VectorType multiplicity;
    scratch_data.initialize_dof_vector(multiplicity, ls_dof_idx);

    const unsigned int dofs_per_cell = scratch_data.get_n_dofs_per_cell(ls_dof_idx);

    FEValues<dim> distance_eval(
      scratch_data.get_mapping(),
      scratch_data.get_dof_handler(ls_dof_idx).get_fe(),
      Quadrature<dim>(scratch_data.get_dof_handler(ls_dof_idx).get_fe().get_unit_support_points()),
      update_values);

    std::vector<number> distance_at_q(distance_eval.n_quadrature_points);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell : scratch_data.get_dof_handler(ls_dof_idx).active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            cell->get_dof_indices(local_dof_indices);

            distance_eval.reinit(cell);

            distance_eval.get_function_values(signed_distance_solver->get_signed_distance(),
                                              distance_at_q);

            const number epsilon_cell = reinit_data.compute_interface_thickness_parameter_epsilon(
              cell->diameter() / std::sqrt(dim) / reinit_data.fe.get_n_subdivisions());

            Vector<number> level_set_local(dofs_per_cell);
            Vector<number> multiplicity_local(dofs_per_cell);

            for (const auto q : distance_eval.quadrature_point_indices())
              {
                multiplicity_local[q] = 1;
                level_set_local[q] =
                  CharacteristicFunctions::tanh_characteristic_function(distance_at_q[q],
                                                                        epsilon_cell);
              }

            scratch_data.get_constraint(ls_dof_idx)
              .distribute_local_to_global(level_set_local, local_dof_indices, level_set);
            scratch_data.get_constraint(ls_dof_idx)
              .distribute_local_to_global(multiplicity_local, local_dof_indices, multiplicity);
          }
      }

    multiplicity.compress(VectorOperation::add);
    level_set.compress(VectorOperation::add);
    /*
     * average the nodally assembled values
     */
    for (unsigned int i = 0; i < multiplicity.locally_owned_size(); ++i)
      if (multiplicity.local_element(i) > 1.0)
        level_set.local_element(i) /= multiplicity.local_element(i);

    scratch_data.get_constraint(ls_dof_idx).distribute(level_set);

    // update ghost values of solution vector
    level_set.update_ghost_values();
  }

  template <int dim, typename number>
  const typename ReinitializationGeometricOperation<dim, number>::VectorType &
  ReinitializationGeometricOperation<dim, number>::get_level_set() const
  {
    if (reinit_data.geometric.transform_to_tanh)
      return level_set;
    else
      return signed_distance_solver->get_signed_distance();
  }

  template <int dim, typename number>
  typename ReinitializationGeometricOperation<dim, number>::VectorType &
  ReinitializationGeometricOperation<dim, number>::get_level_set()
  {
    if (reinit_data.geometric.transform_to_tanh)
      return level_set;
    else
      return signed_distance_solver->get_signed_distance();
  }

  template <int dim, typename number>
  void
  ReinitializationGeometricOperation<dim, number>::attach_vectors(std::vector<VectorType *> &)
  {
    // none
  }

  template <int dim, typename number>
  void
  ReinitializationGeometricOperation<dim, number>::attach_output_vectors(
    GenericDataOut<dim, number> &data_out) const
  {
    data_out.add_data_vector(scratch_data.get_dof_handler(ls_dof_idx),
                             signed_distance_solver->get_signed_distance(),
                             "signed_distance");
    data_out.add_data_vector(scratch_data.get_dof_handler(ls_dof_idx),
                             get_level_set(),
                             "level_set");
  }

  template class ReinitializationGeometricOperation<1, double>;
  template class ReinitializationGeometricOperation<2, double>;
  template class ReinitializationGeometricOperation<3, double>;
} // namespace MeltPoolDG::LevelSet
