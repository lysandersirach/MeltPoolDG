#pragma once
#include <deal.II/base/function.h>
#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/point.h>
#include <deal.II/base/table_handler.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/vector.h>

#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>

#include <meltpooldg/core/simulation_base.hpp>
#include <meltpooldg/utilities/characteristic_functions.hpp>
#include <meltpooldg/utilities/create_triangulation_with_marching_cube_algorithm.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "../level_set_case.hpp"

namespace MeltPoolDG::Simulation::VortexBubble
{
  // period of the vortex flow
  static double Tf = 2.0;

  /*
   * this function specifies the initial field of the level set equation
   */
  template <int dim, typename number>
  class InitializePhi : public dealii::Function<dim, number>
  {
  public:
    InitializePhi()
      : dealii::Function<dim, number>()
      , distance_sphere(dim == 1 ? dealii::Point<dim, number>(0.5) :
                                   dealii::Point<dim, number>(0.5, 0.75),
                        0.15)
    {}

    number
    value(const dealii::Point<dim, number> &p, const unsigned int /*component*/) const override
    {
      return -distance_sphere.value(p);
    }

  private:
    const dealii::Functions::SignedDistance::Sphere<dim> distance_sphere;
  };

  template <int dim, typename number>
  class PhiExact : public dealii::Function<dim, number>
  {
  public:
    PhiExact(const number eps)
      : eps(eps)
    {}

    number
    value(const dealii::Point<dim, number> &p, const unsigned int /*component*/) const override
    {
      return std::tanh(distance_sphere.value(p, 0) / (2 * eps));
    }

  private:
    const number                     eps;
    const InitializePhi<dim, number> distance_sphere;
  };

  template <int dim, typename number>
  class SignedDistanceExact : public dealii::Function<dim, number>
  {
  public:
    SignedDistanceExact(const number eps)
      : eps(eps)
    {}

    number
    value(const dealii::Point<dim, number> &p, const unsigned int /*component*/) const override
    {
      const auto val = distance_sphere.value(p, 0);
      return val > 0 ? std::min(val, 8 * eps) : std::max(val, -8 * eps);
    }

  private:
    const number                     eps;
    const InitializePhi<dim, number> distance_sphere;
  };

  template <int dim, typename number>
  class AdvectionField : public dealii::Function<dim, number>
  {
  public:
    AdvectionField()
      : dealii::Function<dim, number>(dim)
    {}

    void
    vector_value(const dealii::Point<dim, number> &p, dealii::Vector<number> &values) const override
    {
      if constexpr (dim == 2)
        {
          const number time = this->get_time();

          const number x = p[0];
          const number y = p[1];

          const number reverseCoefficient = std::cos(dealii::numbers::PI * time / Tf);

          values[0] = reverseCoefficient * (std::sin(2. * dealii::numbers::PI * y) *
                                            std::pow(std::sin(dealii::numbers::PI * x), 2.));
          values[1] = reverseCoefficient * (-std::sin(2. * dealii::numbers::PI * x) *
                                            std::pow(std::sin(dealii::numbers::PI * y), 2.));
        }
      else
        AssertThrow(false, dealii::ExcMessage("Advection field for dim!=2 not implemented"));
    }
  };

  /* for constant Dirichlet conditions we could also use the ConstantFunction
   * utility from dealii
   */
  template <int dim, typename number>
  class DirichletCondition : public dealii::Function<dim, number>
  {
  public:
    DirichletCondition()
      : dealii::Function<dim, number>()
    {}

    number
    value(const dealii::Point<dim, number> &p, const unsigned int component = 0) const override
    {
      (void)p;
      (void)component;
      return -1.0;
    }
  };

  /*
   *      This class collects all relevant input data for the level set simulation
   */

  template <int dim, typename number>
  class SimulationVortexBubble : public LevelSet::LevelSetCase<dim, number>
  {
  public:
    SimulationVortexBubble(std::string parameter_file, const MPI_Comm mpi_communicator)
      : LevelSet::LevelSetCase<dim, number>(parameter_file, mpi_communicator)
    {}

    void
    create_spatial_discretization() override
    {
      if (dim == 1 || this->parameters.base.fe.type == FiniteElementType::FE_SimplexP)
        {
          AssertDimension(dealii::Utilities::MPI::n_mpi_processes(this->mpi_communicator), 1);
          this->triangulation = std::make_shared<dealii::Triangulation<dim>>();
        }
      else
        {
          this->triangulation = std::make_shared<dealii::parallel::distributed::Triangulation<dim>>(
            this->mpi_communicator);
        }

      if (this->parameters.base.fe.type == FiniteElementType::FE_SimplexP)
        {
          dealii::GridGenerator::subdivided_hyper_cube_with_simplices(
            *this->triangulation,
            dealii::Utilities::pow(2, this->parameters.base.global_refinements),
            left_domain,
            right_domain);
        }
      else
        {
          dealii::GridGenerator::hyper_cube(*this->triangulation, left_domain, right_domain);
          this->triangulation->refine_global(this->parameters.base.global_refinements);
        }
    }

    void
    set_boundary_conditions() override
    {
      // none
    }

    void
    set_field_conditions() override
    {
      this->attach_initial_condition(std::make_shared<InitializePhi<dim, number>>(),
                                     "signed_distance");
      this->attach_field_function(std::make_shared<AdvectionField<dim, number>>(),
                                  "prescribed_velocity",
                                  "level_set");
    }

    void
    do_postprocessing(const GenericDataOut<dim, number> &generic_data_out) const final
    {
      if constexpr (dim == 2)
        {
          dealii::ConditionalOStream pcout(
            std::cout, dealii::Utilities::MPI::this_mpi_process(this->mpi_communicator) == 0);
          // compute area
          const auto        n_q_points = this->parameters.base.fe.degree + 3;
          dealii::FE_Q<dim> fe(this->parameters.base.fe.degree);

          dealii::QGauss<dim>   quadrature(n_q_points);
          dealii::FEValues<dim> fe_values(fe,
                                          quadrature,
                                          dealii::update_values | dealii::update_JxW_values |
                                            dealii::update_quadrature_points);

          std::vector<number> phi_at_q(dealii::QGauss<dim>(n_q_points).size());

          std::vector<number> volume_fraction;
          number              area_droplet = 0;
          number              area_bulk    = 0;
          const number        threshhold   = 0.0;

          generic_data_out.get_vector("level_set").update_ghost_values();

          for (const auto &cell :
               generic_data_out.get_dof_handler("level_set").active_cell_iterators())
            if (cell->is_locally_owned())
              {
                fe_values.reinit(cell);
                fe_values.get_function_values(generic_data_out.get_vector("level_set"),
                                              phi_at_q); // compute values of old solution

                for (const unsigned int q_index : fe_values.quadrature_point_indices())
                  {
                    if (phi_at_q[q_index] >= threshhold)
                      area_droplet += fe_values.JxW(q_index);
                    else
                      area_bulk += fe_values.JxW(q_index);
                  }
              }
          generic_data_out.get_vector("level_set").zero_out_ghost_values();

          area_droplet = dealii::Utilities::MPI::sum(area_droplet, this->mpi_communicator);
          area_bulk    = dealii::Utilities::MPI::sum(area_bulk, this->mpi_communicator);

          pcout << "area (phi>0) " << area_droplet << std::endl;
          pcout << "area (phi<0) " << area_bulk << std::endl;

          // compute circularity
          // 1) compute the surface of the droplet
          dealii::Triangulation<std::max(1, dim - 1), dim> tria_droplet_surface;

          GridGenerator::create_triangulation_with_marching_cube_algorithm<dim>(
            tria_droplet_surface,
            dealii::MappingQ<dim>(3),
            generic_data_out.get_dof_handler("level_set"),
            generic_data_out.get_vector("level_set"),
            0. /*iso level*/,
            1 /*n subdivisions of surface mesh*/);

          number area_droplet_boundary = 0;

          // check if partitioned domains contain surface elements in case of parallel execution
          if (tria_droplet_surface.n_cells() > 0)
            area_droplet_boundary =
              dealii::GridTools::volume<std::max(1, dim - 1), dim>(tria_droplet_surface);

          area_droplet_boundary =
            dealii::Utilities::MPI::sum(area_droplet_boundary, MPI_COMM_WORLD);

          AssertThrow(area_droplet_boundary >= 1e-16,
                      dealii::ExcMessage("Area of droplet is zero."));

          // 2) compute circularity
          number circularity =
            2. * std::sqrt(dealii::numbers::PI * area_droplet) / (area_droplet_boundary);
          pcout << "circularity " << circularity << std::endl;

          table.add_value("time", generic_data_out.get_time());
          table.add_value("circularity", circularity);
          table.add_value("area_droplet", area_droplet);

          if (dealii::Utilities::MPI::this_mpi_process(this->mpi_communicator) == 0)
            {
              namespace fs = std::filesystem;
              std::ofstream output(fs::path(this->parameters.output.directory) /
                                   fs::path(this->parameters.output.paraview.filename + ".txt"));
              table.write_text(output);
            }

          if (std::abs(generic_data_out.get_time() - this->parameters.time_stepping.end_time) <
                1e-10 ||
              (std::abs(generic_data_out.get_time() - this->parameters.time_stepping.start_time) <
               1e-10))
            {
              namespace fs = std::filesystem;
              std::ofstream output(
                fs::path(this->parameters.output.directory) /
                fs::path(this->parameters.output.paraview.filename + "_L2norm.txt"));


              generic_data_out.get_vector("distance").update_ghost_values();
              generic_data_out.get_vector("level_set").update_ghost_values();

              const number eps =
                this->parameters.ls.reinit.compute_interface_thickness_parameter_epsilon(
                  dealii::GridTools::minimal_cell_diameter(*this->triangulation) /
                  this->parameters.ls.get_n_subdivisions() / std::sqrt(dim));

              // compute L2Norm of level_set
              dealii::Vector<float> difference_per_cell(this->triangulation->n_active_cells());
              dealii::VectorTools::integrate_difference(generic_data_out.get_mapping(),
                                                        generic_data_out.get_dof_handler(
                                                          "level_set"),
                                                        generic_data_out.get_vector("level_set"),
                                                        PhiExact<dim, number>(eps),
                                                        difference_per_cell,
                                                        quadrature,
                                                        dealii::VectorTools::L2_norm);

              const number norm =
                dealii::VectorTools::compute_global_error(*this->triangulation,
                                                          difference_per_cell,
                                                          dealii::VectorTools::L2_norm);
              difference_per_cell = 0;

              // compute L2Norm of distance
              dealii::VectorTools::integrate_difference(generic_data_out.get_mapping(),
                                                        generic_data_out.get_dof_handler(
                                                          "level_set"),
                                                        generic_data_out.get_vector("distance"),
                                                        SignedDistanceExact<dim, number>(eps),
                                                        difference_per_cell,
                                                        quadrature,
                                                        dealii::VectorTools::L2_norm);
              const number norm_sd =
                dealii::VectorTools::compute_global_error(*this->triangulation,
                                                          difference_per_cell,
                                                          dealii::VectorTools::L2_norm);

              // compute relative L2 norm in a finite interval around the interface
              SignedDistanceExact<dim, number> sd_circle(eps);
              dealii::FEValues<dim>            fe(generic_data_out.get_mapping(),
                                       generic_data_out.get_dof_handler("level_set").get_fe(),
                                       quadrature,
                                       dealii::update_values | dealii::update_JxW_values |
                                         dealii::update_quadrature_points);

              const unsigned int  n_q_points = fe.get_quadrature().size();
              std::vector<number> phi_at_q(n_q_points);

              number norm_interface_region       = 0;
              number norm_interface_region_exact = 0;
              number norm_interface_kreiss       = 0;

              for (const auto &cell :
                   generic_data_out.get_dof_handler("level_set").active_cell_iterators())
                {
                  if (cell->is_locally_owned())
                    {
                      fe.reinit(cell);
                      fe.get_function_values(generic_data_out.get_vector("distance"), phi_at_q);

                      for (const auto &q : fe.quadrature_point_indices())
                        {
                          norm_interface_kreiss +=
                            dealii::Utilities::fixed_power<2>(
                              CharacteristicFunctions::heaviside(phi_at_q[q]) -
                              CharacteristicFunctions::heaviside(
                                sd_circle.value(fe.quadrature_point(q), 0))) *
                            fe.JxW(q);
                          if (std::abs(phi_at_q[q]) < 0.02)
                            {
                              norm_interface_region +=
                                dealii::Utilities::fixed_power<2>(
                                  sd_circle.value(fe.quadrature_point(q), 0) - phi_at_q[q]) *
                                fe.JxW(q);
                              norm_interface_region_exact +=
                                dealii::Utilities::fixed_power<2>(
                                  sd_circle.value(fe.quadrature_point(q), 0)) *
                                fe.JxW(q);
                            }
                        }
                    }
                }

              norm_interface_region_exact =
                std::sqrt(dealii::Utilities::MPI::sum(norm_interface_region_exact, MPI_COMM_WORLD));
              norm_interface_region =
                std::sqrt(dealii::Utilities::MPI::sum(norm_interface_region, MPI_COMM_WORLD));
              norm_interface_kreiss =
                std::sqrt(dealii::Utilities::MPI::sum(norm_interface_kreiss, MPI_COMM_WORLD));

              generic_data_out.get_vector("level_set").zero_out_ghost_values();
              generic_data_out.get_vector("distance").zero_out_ghost_values();

              if (dealii::Utilities::MPI::this_mpi_process(this->mpi_communicator) == 0)
                output
                  << "n_dofs L2_norm(phi-phiExact) L2_norm(signed_distance-exact) relL2_normGamma(signed_distance-exact) L2_normGamma(sd_exact) L2(Kreiss)"
                  << std::endl
                  << generic_data_out.get_dof_handler("level_set").n_dofs() << " " << norm << " "
                  << norm_sd << " " << norm_interface_region / norm_interface_region_exact << " "
                  << norm_interface_region_exact << " " << norm_interface_kreiss;
            }
        }
    }

    bool
    add_simulation_specific_parameters(dealii::ParameterHandler &prm) override
    {
      prm.enter_subsection("simulation specific");
      {
        prm.add_parameter("T", Tf, "Period of vortex flow.");
      }
      prm.leave_subsection();

      return this->parameters.base.do_print_parameters;
    }

  private:
    number                       left_domain  = 0.0;
    number                       right_domain = 1.0;
    mutable dealii::TableHandler table;
  };
} // namespace MeltPoolDG::Simulation::VortexBubble
