#pragma once
#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>
#include <deal.II/base/types.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools_geometry.h>
#include <deal.II/grid/tria.h>

#include <meltpooldg/core/finite_element_data.hpp>
#include <meltpooldg/core/simulation_base.hpp>
#include <meltpooldg/utilities/characteristic_functions.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "../../melt_pool_case.hpp"

/**
 * This example is derived from
 *
 * Nas, S., & Tryggvason, G. (2003). Thermocapillary interaction of two bubbles or drops.
 * International Journal of Multiphase Flow, 29(7), 1117-1135.
 * https://doi.org/10.1016/S0301-9322(03)00084-3
 *
 * and
 *
 * Balcázar, N., Rigola, J., Castro, J., & Oliva, A. (2016). A level-set model for thermocapillary
 * motion of deformable fluid particles. International Journal of Heat and Fluid Flow, 62, 324-343.
 * https://doi.org/10.1016/j.ijheatfluidflow.2016.09.015
 *
 * In contrast to the papers' cases the two droplets can coalesce here.
 *
 *                 T2 = fixed
 *                  no slip
 *            +----------------------------------------+                    -
 *            |                                        |                    |
 *            |                                        |                    |
 *            |                                        |                    |
 *            |                        -----           |                    |
 *            |                       --   --          |                    |
 *            |                      --     --         |                    |
 *            |                       --   --          |                   16a
 *            |          -----         -----           |                    |
 *   periodic |         --   --          |-a-|         |  periodic          |
 *            |        --     --                       |                    |
 *            |         --   --                        |                    |
 *            |          -----                         |                    |
 *            |                                        |                    |
 *            |                                        |                    |
 *            +----------------------------------------+                    -
 *                   no slip
 *                  T1 = fixed
 *
 *            |----------------- 8a --------------------|
 *
 * droplet radius: a = 0.048 m
 * droplet positions: (2.9a, 4a), (5.1a, 5.8a)
 *
 * characteristics:
 *    Re = 40
 *    Ma = 40
 *    Ca = 0.041666
 *
 * droplet:
 *    rho_i    = 250 kg/m³
 *    mu_i     = 0.012  N/m²s
 *    lambda_i = 1.2e-6 W/m/K
 *    cp_i     = 5e-5 J/kg/K
 *
 * ambient fluid:
 *    rho_0    = 500 kg/m³
 *    mu_0     = 0.024 N/m²s
 *    lambda_0 = 2.4e-6 W/m/K
 *    cp_0     = 1e-4 J/kg/K
 *
 * surface tension coefficient:
 *    sigma_0 = 0.023040369 N/m
 * temperature-dependent surface tension coefficient:
 *    sigma_T = 0.002 N/m/K
 *
 * temperature:
 *    ∇T = 10 K/m
 *    T1 = 0 K
 *    T2 = 0 + 16 * a * ∇T = 7.68 K
 *
 * reference velocity
 *    Ur = sigma_T * a * ∇T / mu_0 = 0.04 m/s
 * reference time scale
 *    tr = a / Ur = 1.2 s
 *
 * time step size
 *    Δt = 0.1 * min( a / Ur , a^1.5 * ( (rho_0 + rho_i)/(4 * pi * sigma_0) )^0.5 ) = 0.05 s
 */

namespace MeltPoolDG::Simulation::ThermoCapillaryTwoDroplets
{
  using namespace MeltPoolDG::Simulation;

  static constexpr double radius                = 0.048;
  static constexpr double x_outer               = 8. * radius;
  static constexpr double z_outer               = 16. * radius;
  static constexpr double reference_temperature = 0.0;
  static constexpr double temperature_gradient  = 10.;

  template <int dim>
  class InitialValuesLS : public dealii::Function<dim>
  {
  public:
    InitialValuesLS(const double eps)
      : dealii::Function<dim>()
      , eps(eps)
      , center1(dim == 2 ? dealii::Point<dim>(2.9 * radius, 4.0 * radius) :
                           dealii::Point<dim>(2.9 * radius, 0, 4.0 * radius))

      , center2(dim == 2 ? dealii::Point<dim>(5.1 * radius, 5.8 * radius) :
                           dealii::Point<dim>(5.1 * radius, 0, 5.8 * radius))
      , sphere1(center1, radius)
      , sphere2(center2, radius)
    {}

    double
    value(const dealii::Point<dim> &p, const unsigned int /*component*/) const override
    {
      if ((p - center1).norm() < (p - center2).norm()) // closer to first droplet center
        return CharacteristicFunctions::tanh_characteristic_function(-sphere1.value(p), eps);
      else // closer to second droplet center
        return CharacteristicFunctions::tanh_characteristic_function(-sphere2.value(p), eps);
    }

    const double                                         eps;
    const dealii::Point<dim>                             center1;
    const dealii::Point<dim>                             center2;
    const dealii::Functions::SignedDistance::Sphere<dim> sphere1;
    const dealii::Functions::SignedDistance::Sphere<dim> sphere2;
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
      return reference_temperature + p[dim - 1] * temperature_gradient;
    }
  };
  /*
   *      This class collects all relevant input data for the level set simulation
   */

  template <int dim, typename number>
  class SimulationThermoCapillaryTwoDroplets : public MeltPoolCase<dim, number>
  {
  public:
    SimulationThermoCapillaryTwoDroplets(std::string    parameter_file,
                                         const MPI_Comm mpi_communicator)
      : MeltPoolCase<dim, number>(parameter_file, mpi_communicator)
    {}

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
            dim == 2 ? dealii::Point<dim>(0, 0) : dealii::Point<dim>(0, 0, 0);
          const dealii::Point<dim> top_right = dim == 2 ? dealii::Point<dim>(x_outer, z_outer) :
                                                          dealii::Point<dim>(x_outer, 0, z_outer);

          // create mesh
          if (this->parameters.base.fe.type == FiniteElementType::FE_SimplexP)
            {
              std::vector<unsigned int> subdivisions(
                dim, 4 * dealii::Utilities::pow(2, this->parameters.base.global_refinements));
              subdivisions[dim - 1] *= 2;
              dealii::GridGenerator::subdivided_hyper_rectangle_with_simplices(*this->triangulation,
                                                                               subdivisions,
                                                                               bottom_left,
                                                                               top_right);
            }
          else
            {
              std::vector<unsigned int> subdivisions(dim, 4);
              subdivisions[dim - 1] *= 2;
              dealii::GridGenerator::subdivided_hyper_rectangle(*this->triangulation,
                                                                subdivisions,
                                                                bottom_left,
                                                                top_right);
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

      if constexpr (dim == 2)
        {
          for (const auto &cell : this->triangulation->cell_iterators())
            for (const auto &face : cell->face_iterators())
              if (face->at_boundary())
                {
                  if (face->center()[1] == 0.0)
                    face->set_boundary_id(lower_bc);
                  else if (face->center()[1] == z_outer)
                    face->set_boundary_id(upper_bc);
                  else if (face->center()[0] == 0.0)
                    face->set_boundary_id(left_bc);
                  else if (face->center()[0] == x_outer)
                    face->set_boundary_id(right_bc);
                }
        }
      else
        {
          AssertThrow(false, dealii::ExcNotImplemented());
        }

      this->attach_boundary_condition(lower_bc, "fix_pressure_constant", "navier_stokes_p");
      this->attach_boundary_condition(lower_bc, "no_slip", "navier_stokes_u");
      this->attach_boundary_condition(upper_bc, "no_slip", "navier_stokes_u");
      this->attach_periodic_boundary_condition(left_bc, right_bc, 0);

      this->attach_boundary_condition({lower_bc, std::make_shared<InitialValuesTemperature<dim>>()},
                                      "dirichlet",
                                      "heat_transfer");
      this->attach_boundary_condition({upper_bc, std::make_shared<InitialValuesTemperature<dim>>()},
                                      "dirichlet",
                                      "heat_transfer");


      if (this->parameters.base.fe.type != FiniteElementType::FE_SimplexP)
        this->triangulation->refine_global(this->parameters.base.global_refinements);
    }

    void
    set_field_conditions() final
    {
      const double eps = this->parameters.ls.reinit.compute_interface_thickness_parameter_epsilon(
        dealii::GridTools::minimal_cell_diameter(*this->triangulation) /
        this->parameters.ls.get_n_subdivisions() / std::sqrt(dim));

      AssertThrow(eps > 0, dealii::ExcNotImplemented());

      this->attach_initial_condition(std::make_shared<InitialValuesLS<dim>>(eps), "level_set");
      this->attach_initial_condition(std::make_shared<dealii::Functions::ZeroFunction<dim>>(dim),
                                     "navier_stokes_u");
      this->attach_initial_condition(std::make_shared<InitialValuesTemperature<dim>>(),
                                     "heat_transfer");
    }
  };
} // namespace MeltPoolDG::Simulation::ThermoCapillaryTwoDroplets
