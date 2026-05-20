#pragma once

#include <deal.II/base/function.h>
#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>

#include <meltpooldg/core/simulation_base.hpp>
#include <meltpooldg/utilities/characteristic_functions.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>

#include <iostream>

#include "../../melt_pool_case.hpp"

namespace MeltPoolDG::Simulation::EvaporatingDropletWithHeat
{
  template <int dim>
  class InitialValuesLS : public dealii::Function<dim>
  {
  public:
    InitialValuesLS(const double eps)
      : dealii::Function<dim>()
      , distance_sphere(dim == 2 ? dealii::Point<dim>(0, 0) : dealii::Point<dim>(0, 0, 0), 0.02)
      , eps(eps)
    {}

    double
    value(const dealii::Point<dim> &p, const unsigned int /*component*/) const override
    {
      return CharacteristicFunctions::tanh_characteristic_function(-distance_sphere.value(p), eps);
    }

  private:
    const dealii::Functions::SignedDistance::Sphere<dim> distance_sphere;
    const double                                         eps;
  };

  /*
   *      This class collects all relevant input data for the level set simulation
   */

  template <int dim, typename number>
  class SimulationEvaporatingDropletWithHeat : public MeltPoolCase<dim, number>
  {
  public:
    SimulationEvaporatingDropletWithHeat(std::string    parameter_file,
                                         const MPI_Comm mpi_communicator)
      : MeltPoolCase<dim, number>(parameter_file, mpi_communicator)
      , lambda(2. * dealii::numbers::PI *
               std::sqrt(3. * this->parameters.flow.surface_tension.surface_tension_coefficient /
                         (9.81 * (this->parameters.material.liquid.density -
                                  this->parameters.material.gas.density))))
    {}

    void
    create_spatial_discretization() override
    {
      if (this->parameters.base.fe.type == FiniteElementType::FE_SimplexP)
        {
          this->triangulation =
            std::make_shared<dealii::parallel::shared::Triangulation<dim>>(this->mpi_communicator);
        }
      else
        {
          this->triangulation = std::make_shared<dealii::parallel::distributed::Triangulation<dim>>(
            this->mpi_communicator);
        }

      if constexpr ((dim == 2) || (dim == 3))
        {
          // create mesh
          std::vector<unsigned int> subdivisions(
            dim,
            5 * (this->parameters.base.fe.type == FiniteElementType::FE_SimplexP ?
                   dealii::Utilities::pow(2, this->parameters.base.global_refinements) :
                   1));

          const dealii::Point<dim> top_right =
            (dim == 2) ? dealii::Point<dim>(lambda / 2., lambda / 2.) :
                         dealii::Point<dim>(lambda / 2., lambda / 2., lambda / 2.);
          const dealii::Point<dim> bottom_left =
            (dim == 2) ? dealii::Point<dim>(-lambda / 2., -lambda / 2.) :
                         dealii::Point<dim>(-lambda / 2., -lambda / 2., -lambda / 2.);

          if (this->parameters.base.fe.type == FiniteElementType::FE_SimplexP)
            {
              dealii::GridGenerator::subdivided_hyper_rectangle_with_simplices(*this->triangulation,
                                                                               subdivisions,
                                                                               bottom_left,
                                                                               top_right);
            }
          else
            {
              dealii::GridGenerator::subdivided_hyper_rectangle(*this->triangulation,
                                                                subdivisions,
                                                                bottom_left,
                                                                top_right);
              this->triangulation->refine_global(this->parameters.base.global_refinements);
            }
        }
      else
        {
          AssertThrow(false, dealii::ExcNotImplemented());
        }
    }

    void
    set_boundary_conditions() override
    {
      auto dirichlet_temp = std::make_shared<dealii::Functions::ConstantFunction<dim>>(10);

      this->attach_boundary_condition({0, dirichlet_temp}, "dirichlet", "heat_transfer");
      this->attach_boundary_condition({0, std::make_shared<dealii::Functions::ZeroFunction<dim>>()},
                                      "open",
                                      "navier_stokes_u");
    }

    void
    set_field_conditions() override
    {
      const double eps = this->parameters.ls.reinit.compute_interface_thickness_parameter_epsilon(
        dealii::GridTools::minimal_cell_diameter(*this->triangulation) /
        this->parameters.ls.get_n_subdivisions() / std::sqrt(dim));

      AssertThrow(eps > 0, dealii::ExcNotImplemented());

      this->attach_initial_condition(std::make_shared<InitialValuesLS<dim>>(eps), "level_set");
      this->attach_initial_condition(std::shared_ptr<dealii::Function<dim>>(
                                       new dealii::Functions::ZeroFunction<dim>(dim)),
                                     "navier_stokes_u");
      this->attach_initial_condition(std::shared_ptr<dealii::Function<dim>>(
                                       new dealii::Functions::ConstantFunction<dim>(0.0)),
                                     "heat_transfer");
    }

    const double lambda;
  };
} // namespace MeltPoolDG::Simulation::EvaporatingDropletWithHeat
