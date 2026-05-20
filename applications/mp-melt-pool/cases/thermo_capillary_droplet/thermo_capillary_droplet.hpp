#pragma once

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/mpi_remote_point_evaluation.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/patterns.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/types.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools_geometry.h>
#include <deal.II/grid/tria.h>

#include <deal.II/numerics/vector_tools_evaluate.h>

#include <meltpooldg/core/finite_element_data.hpp>
#include <meltpooldg/core/simulation_base.hpp>
#include <meltpooldg/flow/characteristic_numbers.hpp>
#include <meltpooldg/post_processing/generic_data_out.hpp>
#include <meltpooldg/utilities/characteristic_functions.hpp>

#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

#include "../../melt_pool_case.hpp"

/**
 * This example is derived from
 *
 * Meier et al. 2020
 * "A novel smoothed particle hydrodynamics formulation
 * for thermo-capillary phase change problems with focus on metal additive
 * manufacturing melt pool modeling"
 *
 * and
 *
 * C. Ma, D. Bothe 2011, "Direct numerical simulation of thermocapillary flow
 * based on the Volume of Fluid method"
 *
 *                 T2 = fixed
 *                  no slip
 *            +-------------------+
 *            |                   |
 *            |                   |
 *            |       -----       |
 *   sym      |      --   --      |   sym
 * T Neumann  |     --     --     |   T Neumann
 *            |      --   --      |
 *            |       -----       |
 *            |                   |
 *            +-------------------+
 *                   no slip
 *                  T1 = fixed
 *
 * droplet:
 *    rho    = 250 kg/m³
 *    mu     = 0.012 kg/m/s
 *    lambda = 1.2e-6 W/m/K
 *    cp     = 5e-5 J/kg/K
 *
 * ambient fluid:
 *    rho    = 500 kg/m³
 *    mu     = 0.024 N/m²s
 *    lambda = 2.4e-6 W/m/K
 *    cp     = 1e-4 J/kg/K
 *
 * droplet radius: a=1.44e-3 m
 * surface tension coefficient: 0.01 N/m
 * temperature-dependent surface tension coefficient: 0.002 N/m/K
 * T1 = 290 K
 * T2 = 290 + 4 * a * 200 K/m = 291.152 K
 */

namespace MeltPoolDG::Simulation::ThermoCapillaryDroplet
{
  using namespace MeltPoolDG::Simulation;


  static constexpr double a      = 1.44e-3;
  static constexpr double grad_T = 200;

  template <int dim>
  class InitialValuesLS : public dealii::Function<dim>
  {
  public:
    InitialValuesLS(const double eps, const bool liquid_phase_outside)
      : dealii::Function<dim>()
      , distance_sphere(dim == 2 ? dealii::Point<dim>(0, 0) : dealii::Point<dim>(0, 0, 0), a)
      , eps(eps)
      , factor(liquid_phase_outside ? 1.0 : -1.0)
    {}

    double
    value(const dealii::Point<dim> &p, const unsigned int /*component*/) const override
    {
      return CharacteristicFunctions::tanh_characteristic_function(factor *
                                                                     distance_sphere.value(p),
                                                                   eps);
    }

  private:
    const dealii::Functions::SignedDistance::Sphere<dim> distance_sphere;
    const double                                         eps;
    const double                                         factor;
  };


  template <int dim>
  class InitialValuesTemperature : public dealii::Function<dim>
  {
  public:
    InitialValuesTemperature()
      : dealii::Function<dim>()
    {}

    double
    value(const dealii::Point<dim> &p, const unsigned int /*component*/) const override
    {
      return 290 + grad_T * (p[dim - 1] + 2 * a);
    }
  };
  /*
   *      This class collects all relevant input data for the level set simulation
   */

  template <int dim, typename number>
  class SimulationThermoCapillaryDroplet : public MeltPoolCase<dim, number>
  {
  public:
    SimulationThermoCapillaryDroplet(std::string parameter_file, const MPI_Comm mpi_communicator)
      : MeltPoolCase<dim, number>(parameter_file, mpi_communicator)
    {
      if (this->parameters.output.do_user_defined_postprocessing and
          dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
        file.open(this->parameters.output.directory + "/" +
                  this->parameters.output.paraview.filename +
                  "_droplet_velocity_over_time_normalized.csv");
      velocity_reference =
        this->parameters.flow.surface_tension.temperature_dependent_surface_tension_coefficient *
        grad_T * a / this->parameters.material.gas.dynamic_viscosity;
      time_reference = a / velocity_reference;

      const auto characteristic_numbers =
        Flow::CharacteristicNumbers<double>(this->parameters.material.gas);
      reynolds_number  = characteristic_numbers.Reynolds(velocity_reference, a);
      mach_number      = characteristic_numbers.Mach(velocity_reference, a);
      capillary_number = characteristic_numbers.capillary(
        velocity_reference, this->parameters.flow.surface_tension.surface_tension_coefficient);
    }

    bool
    add_simulation_specific_parameters(dealii::ParameterHandler &prm) override
    {
      prm.enter_subsection("simulation specific");
      {
        prm.add_parameter(
          "liquid phase outside",
          liquid_phase_outside,
          "set this parameter to true to flip the level set and have the \"liquid\" phase outside the droplet.",
          dealii::Patterns::Bool());
      }
      prm.leave_subsection();

      return this->parameters.base.do_print_parameters;
    }

    void
    create_spatial_discretization() override
    {
      if (this->parameters.base.fe.type == FiniteElementType::FE_SimplexP)
        {
#ifdef DEAL_II_WITH_METIS
          this->triangulation = std::make_shared<dealii::parallel::shared::Triangulation<dim>>(
            this->mpi_communicator,
            dealii::Triangulation<dim>::none,
            false,
            dealii::parallel::shared::Triangulation<dim>::Settings::partition_metis);
#else
          AssertThrow(
            false,
            dealii::ExcMessage(
              "Missing Metis support of the deal.II installation. "
              "Configure deal.II with -D DEAL_II_WITH_METIS='ON' to execute this example."));
#endif
        }
      else
        {
          this->triangulation = std::make_shared<dealii::parallel::distributed::Triangulation<dim>>(
            this->mpi_communicator);
        }

      if constexpr (dim == 2 or dim == 3)
        {
          // create mesh
          const dealii::Point<dim> bottom_left =
            dim == 2 ? dealii::Point<dim>(x_min, x_min) : dealii::Point<dim>(x_min, x_min, x_min);
          const dealii::Point<dim> top_right =
            dim == 2 ? dealii::Point<dim>(x_max, x_max) : dealii::Point<dim>(x_max, x_max, x_max);

          if (this->parameters.base.fe.type == FiniteElementType::FE_SimplexP)
            {
              // create mesh
              std::vector<unsigned int> subdivisions(
                dim, 5 * dealii::Utilities::pow(2, this->parameters.base.global_refinements));
              subdivisions[dim - 1] *= 2;

              dealii::GridGenerator::subdivided_hyper_rectangle_with_simplices(*this->triangulation,
                                                                               subdivisions,
                                                                               bottom_left,
                                                                               top_right);
            }
          else
            {
              dealii::GridGenerator::hyper_rectangle(*this->triangulation, bottom_left, top_right);
              this->triangulation->refine_global(this->parameters.base.global_refinements);
            }
        }
      else
        {
          AssertThrow(false, dealii::ExcNotImplemented());
        }
    }

    void
    set_boundary_conditions() final
    {
      /*
       *  create a pair of (boundary_id, dirichlet_function)
       */

      const dealii::types::boundary_id lower_bc = 1;
      const dealii::types::boundary_id upper_bc = 2;
      const dealii::types::boundary_id left_bc  = 3;
      const dealii::types::boundary_id right_bc = 4;

      this->attach_boundary_condition(lower_bc, "no_slip", "navier_stokes_u");
      this->attach_boundary_condition(upper_bc, "no_slip", "navier_stokes_u");

      this->attach_boundary_condition(left_bc, "symmetry", "navier_stokes_u");
      this->attach_boundary_condition(right_bc, "symmetry", "navier_stokes_u");

      this->attach_boundary_condition({lower_bc, std::make_shared<InitialValuesTemperature<dim>>()},
                                      "dirichlet",
                                      "heat_transfer");
      this->attach_boundary_condition({upper_bc, std::make_shared<InitialValuesTemperature<dim>>()},
                                      "dirichlet",
                                      "heat_transfer");

      if constexpr (dim == 2)
        {
          for (const auto &cell : this->triangulation->cell_iterators())
            for (const auto &face : cell->face_iterators())
              {
                if (not face->at_boundary())
                  continue;
                if (face->center()[1] == x_min)
                  face->set_boundary_id(lower_bc);
                else if (face->center()[1] == x_max)
                  face->set_boundary_id(upper_bc);
                else if (face->center()[0] == x_min)
                  face->set_boundary_id(left_bc);
                else if (face->center()[0] == x_max)
                  face->set_boundary_id(right_bc);
              }
        }
      else if constexpr (dim == 3)
        {
          const dealii::types::boundary_id front_bc = 5;
          const dealii::types::boundary_id back_bc  = 6;
          this->attach_boundary_condition(front_bc, "symmetry", "navier_stokes_u");
          this->attach_boundary_condition(back_bc, "symmetry", "navier_stokes_u");
          for (const auto &cell : this->triangulation->cell_iterators())
            for (const auto &face : cell->face_iterators())
              {
                if (not face->at_boundary())
                  continue;
                if (face->center()[1] == x_min)
                  face->set_boundary_id(lower_bc);
                else if (face->center()[1] == x_max)
                  face->set_boundary_id(upper_bc);
                else if (face->center()[0] == x_min)
                  face->set_boundary_id(left_bc);
                else if (face->center()[0] == x_max)
                  face->set_boundary_id(right_bc);
                else if (face->center()[2] == x_min)
                  face->set_boundary_id(back_bc);
                else if (face->center()[2] == x_max)
                  face->set_boundary_id(front_bc);
              }
        }
      else
        {
          AssertThrow(false, dealii::ExcNotImplemented());
        }
    }

    void
    set_field_conditions() final
    {
      const double eps = this->parameters.ls.reinit.compute_interface_thickness_parameter_epsilon(
        dealii::GridTools::minimal_cell_diameter(*this->triangulation) /
        this->parameters.ls.get_n_subdivisions() / std::sqrt(dim));

      this->attach_initial_condition(std::make_shared<InitialValuesLS<dim>>(eps,
                                                                            liquid_phase_outside),
                                     "level_set");
      this->attach_initial_condition(std::make_shared<dealii::Functions::ZeroFunction<dim>>(dim),
                                     "navier_stokes_u");
      this->attach_initial_condition(std::make_shared<InitialValuesTemperature<dim>>(),
                                     "heat_transfer");
    }

    void
    do_postprocessing(const GenericDataOut<dim, double> &generic_data_out) const final
    {
      if (not this->parameters.output.do_user_defined_postprocessing)
        return;

      if constexpr (dim > 1)
        {
          dealii::ConditionalOStream pcout(std::cout,
                                           dealii::Utilities::MPI::this_mpi_process(
                                             this->mpi_communicator) == 0 and
                                             this->parameters.base.verbosity_level > 0);

          const auto &level_set = generic_data_out.get_vector("level_set");
          const auto &velocity  = generic_data_out.get_vector("velocity");
          if (not level_set.has_ghost_elements())
            level_set.update_ghost_values();
          if (not velocity.has_ghost_elements())
            velocity.update_ghost_values();

          const auto &level_set_dof_handler = generic_data_out.get_dof_handler("level_set");
          const auto &velocity_dof_handler  = generic_data_out.get_dof_handler("velocity");

          dealii::Point<dim> center_of_mass;

          double average_velocity = 0.0;
          double area_of_phase    = 0.0;

          dealii::FEValues<dim> level_set_eval(
            generic_data_out.get_mapping(),
            level_set_dof_handler.get_fe(),
            dealii::QGauss<dim>(level_set_dof_handler.get_fe().tensor_degree() + 1),
            dealii::update_quadrature_points | dealii::update_JxW_values | dealii::update_values);

          const dealii::FEValuesExtractors::Vector velocities(0);
          dealii::FEValues<dim>                    vel_eval(generic_data_out.get_mapping(),
                                         velocity_dof_handler.get_fe(),
                                         dealii::QGauss<dim>(
                                           level_set_dof_handler.get_fe().tensor_degree() + 1),
                                         dealii::update_values);

          std::vector<double>                 ls_at_q(level_set_eval.n_quadrature_points);
          std::vector<dealii::Tensor<1, dim>> vel_at_q(level_set_eval.n_quadrature_points,
                                                       dealii::Tensor<1, dim>());

          typename dealii::DoFHandler<dim>::active_cell_iterator vel_cell =
            velocity_dof_handler.begin_active();

          for (const auto &cell : level_set_dof_handler.active_cell_iterators())
            {
              if (cell->is_locally_owned())
                {
                  level_set_eval.reinit(cell);
                  level_set_eval.get_function_values(level_set, ls_at_q);

                  vel_eval.reinit(vel_cell);
                  vel_eval[velocities].get_function_values(velocity, vel_at_q);

                  for (const auto q : level_set_eval.quadrature_point_indices())
                    {
                      if (ls_at_q[q] <= 0.0)
                        continue;
                      area_of_phase += level_set_eval.JxW(q);
                      average_velocity += vel_at_q[q][dim - 1] * level_set_eval.JxW(q);
                      for (unsigned int d = 0; d < dim; ++d)
                        center_of_mass[d] +=
                          level_set_eval.quadrature_point(q)[d] * level_set_eval.JxW(q);
                    }
                }
              ++vel_cell;
            }

          /*
           * area of the phase
           */
          double global_area = dealii::Utilities::MPI::sum(area_of_phase, this->mpi_communicator);

          /*
           * centroid position
           */
          dealii::Point<dim> global_center_of_mass;
          for (unsigned int d = 0; d < dim; ++d)
            global_center_of_mass[d] =
              dealii::Utilities::MPI::sum(center_of_mass[d], this->mpi_communicator);

          global_center_of_mass /= global_area;

          /*
           * average velocity
           */
          double global_average_velocity =
            dealii::Utilities::MPI::sum(average_velocity, this->mpi_communicator);
          global_average_velocity /= global_area;

          /*
           * velocity measured at centroid
           */
          dealii::Utilities::MPI::RemotePointEvaluation<dim, dim> cache;
          std::vector<dealii::Tensor<1, dim>>                     velocity_of_center =
            dealii::VectorTools::point_values<dim>(generic_data_out.get_mapping(),
                                                   velocity_dof_handler,
                                                   velocity,
                                                   {global_center_of_mass},
                                                   cache);

          pcout << "---------------------------------------------" << std::endl;
          pcout << "    user defined postprocessing" << std::endl;
          pcout << "---------------------------------------------" << std::endl;
          if (print_once)
            {
              pcout << "reference velocity: " << velocity_reference << std::endl;
              pcout << "reference time: " << time_reference << std::endl;
              pcout << "Reynolds number: " << reynolds_number << std::endl;
              pcout << "Mach number: " << mach_number << std::endl;
              pcout << "Capillary number: " << capillary_number << std::endl;
            }

          const auto max_vel = velocity.linfty_norm();
          if (file.is_open())
            {
              pcout << "centroid: " << global_center_of_mass << std::endl;
              pcout << "vel: " << velocity_of_center[0][dim - 1] << std::endl;
              if (print_once)
                file << "time,y_center,t/tr,u/ur,u_max/ur,u_avg/ur" << std::endl;

              file << generic_data_out.get_time() << ", " << global_center_of_mass[dim - 1] << ", "
                   << generic_data_out.get_time() / time_reference << ", "
                   << velocity_of_center[0][dim - 1] / velocity_reference << ", "
                   << max_vel / velocity_reference << ", "
                   << global_average_velocity / velocity_reference << std::endl;
            }
          pcout << "---------------------------------------------" << std::endl;
          pcout << "---------------------------------------------" << std::endl;
        }
      print_once = false;
    }

  private:
    const double x_min = -2 * a;
    const double x_max = 2 * a;
    double       velocity_reference;
    double       time_reference;
    double       reynolds_number;
    double       mach_number;
    double       capillary_number;
    bool         liquid_phase_outside = false;
    // postprocessing
    mutable std::ofstream file;
    mutable bool          print_once = true;
  };
} // namespace MeltPoolDG::Simulation::ThermoCapillaryDroplet
