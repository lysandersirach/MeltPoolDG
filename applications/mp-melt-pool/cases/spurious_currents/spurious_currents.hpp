#pragma once

#include <deal.II/base/function.h>
#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>

#include <meltpooldg/core/simulation_base.hpp>
#include <meltpooldg/utilities/functions.hpp>

#include <iostream>

#include "../../melt_pool_case.hpp"

namespace MeltPoolDG::Simulation::SpuriousCurrents
{
  static double side_length = 5.0;
  static double radius      = 0.5;
  static double radius_2    = 0.75;

  template <int dim, typename number>
  class SimulationSpuriousCurrents : public MeltPoolCase<dim, number>
  {
  public:
    SimulationSpuriousCurrents(std::string parameter_file, const MPI_Comm mpi_communicator)
      : MeltPoolCase<dim, number>(parameter_file, mpi_communicator)
    {}

    bool
    add_simulation_specific_parameters(dealii::ParameterHandler &prm) override
    {
      prm.enter_subsection("simulation specific parameters");
      {
        prm.add_parameter("droplet shape",
                          droplet_shape,
                          "Shape of the droplet: circle or ellipse",
                          dealii::Patterns::Selection("circle|ellipse"));
        prm.add_parameter("side length", side_length, "Side length of the quadratic domain.");
        prm.add_parameter("radius",
                          radius,
                          "Radius of (i) circle or (ii) of the first semi-axis of an ellipse.");
        prm.add_parameter("radius 2", radius_2, "Radius of second semi-axis of an ellipse.");
      }
      prm.leave_subsection();

      return this->parameters.base.do_print_parameters;
    }

    void
    create_spatial_discretization() override
    {
      this->triangulation =
        std::make_shared<dealii::parallel::distributed::Triangulation<dim>>(this->mpi_communicator);

      if constexpr (dim == 2)
        {
          dealii::GridGenerator::hyper_cube(*this->triangulation,
                                            -side_length / 2.,
                                            side_length / 2);
          this->triangulation->refine_global(this->parameters.base.global_refinements);
        }
      else
        {
          AssertThrow(false, dealii::ExcNotImplemented());
        }
    }

    void
    set_boundary_conditions() override
    {
      this->attach_boundary_condition(
        {0, std::make_shared<dealii::Functions::ConstantFunction<dim>>(-1)},
        "dirichlet",
        "level_set");
      this->attach_boundary_condition(0, "no_slip", "navier_stokes_u");
      this->attach_boundary_condition(0, "fix_pressure_constant", "navier_stokes_p");
    }

    void
    set_field_conditions() override
    {
      const double eps = this->parameters.ls.reinit.compute_interface_thickness_parameter_epsilon(
        dealii::GridTools::minimal_cell_diameter(*this->triangulation) /
        this->parameters.ls.get_n_subdivisions() / std::sqrt(dim));

      AssertThrow(eps > 0, dealii::ExcNotImplemented());

      // introduce slightly non-symmetric geometry
      dealii::Point<dim> center;
      for (unsigned int d = 0; d < dim; ++d)
        center[d] = 0.02 + 0.01 * d;

      if (droplet_shape == "circle")
        {
          this->attach_initial_condition(
            std::make_shared<Functions::ChangedSignFunction<dim, double>>(
              std::make_shared<dealii::Functions::SignedDistance::Sphere<dim>>(center, radius)),
            "signed_distance");
        }
      else if (droplet_shape == "ellipse")
        {
          std::array<double, dim> radii;
          if constexpr (dim == 2)
            radii = {{radius_2, radius}};
          else
            AssertThrow(false, dealii::ExcNotImplemented());

          this->attach_initial_condition(
            std::make_shared<Functions::ChangedSignFunction<dim, double>>(
              std::make_shared<dealii::Functions::SignedDistance::Ellipsoid<dim>>(center, radii)),
            "signed_distance");
        }
      else
        AssertThrow(false,
                    dealii::ExcMessage("Unknown droptlet shape: \"" + droplet_shape +
                                       "\"! Abort..."));
      this->attach_initial_condition(std::make_shared<dealii::Functions::ZeroFunction<dim>>(dim),
                                     "navier_stokes_u");
    }

  private:
    std::string droplet_shape = "circle";
  };
} // namespace MeltPoolDG::Simulation::SpuriousCurrents
