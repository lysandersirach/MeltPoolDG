#pragma once
#include <deal.II/base/function.h>
#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/point.h>
#include <deal.II/base/table_handler.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/numerics/vector_tools.h>

#include <meltpooldg/level_set/level_set_tools.hpp>
#include <meltpooldg/utilities/boundary_ids_colorized.hpp>
#include <meltpooldg/utilities/characteristic_functions.hpp>
#include <meltpooldg/utilities/numbers.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "../../reinitialization_case.hpp"

namespace MeltPoolDG::Simulation::WallWetting
{
  /**
   * @brief Initial condition of the level-set field.
   *
   * @tparam dim Number of spatial dimensions of the simulation
   * @tparam number Numeric type used for computations (e.g., float, double)
   */
  template <int dim, typename number>
  class InitialLevelSet : public dealii::Function<dim>
  {
  public:
    /**
     * @brief Function describing the initial state of the level-set field. The function computes a
     * field with the interface as a vertical at a given position x = @p p_x_interface.
     *
     * @param[in] p_x_interface Position of the vertical interface (\f$ phi \f$ = 0).
     * @param[in] p_epsilon Interface thickness parameter.
     */
    InitialLevelSet(const number p_x_interface, const number p_epsilon)
      : dealii::Function<dim>()
      , x_interface(p_x_interface)
      , epsilon(p_epsilon)
    {}

    /**
     * @brief Evaluate the value of the level-set indicator (\f$ phi \f$) at a given point @p p.
     *
     * @param[in] p Evaluation point coordinates
     *
     * @return Value of the level-set indicator
     */
    double
    value(const dealii::Point<dim> &p, const unsigned int /*component*/) const final
    {
      return -CharacteristicFunctions::tanh_characteristic_function(x_interface - p[0], epsilon);
    }

  private:
    /// Position of the vertical interface
    number x_interface;
    /// Interface thickness parameter
    number epsilon;
  };

  /**
   * @brief Function that imposes the bottom wall wetting boundary condition component-wise for
   * component-wise AffineConstraints.
   *
   * @tparam dim Number of spatial dimensions of the simulation
   * @tparam number Numeric type used for computations (e.g., float, double)
   */
  template <int dim, typename number>
  class BottomBoundaryNormalPerComponent : public dealii::Function<dim>
  {
  public:
    /**
     * @brief Function used to compute the normal vector at the bottom boundary considering wetting.
     *
     * @paramp[in] p_vector_component Component of the normal vector to compute
     *
     * @paramp[in] p_contact_angle Imposed static contact angle value in radians
     */
    BottomBoundaryNormalPerComponent(const number p_vector_component, const number p_contact_angle)
      : dealii::Function<dim>()
      , vector_component(p_vector_component)
      , contact_angle(p_contact_angle)
    {}

    /**
     * @brief Evaluate the value of the normal vector component at a given point of the boundary.
     *
     * @return Value of the normal vector component
     */
    double
    value(const dealii::Point<dim> & /*p*/, const unsigned int /*component*/) const final
    {
      if (vector_component == 0)
        return std::sin(contact_angle);
      else /*component == 1*/
        return std::cos(contact_angle);
    }

  private:
    /// Component of the normal vector currently evaluated
    const number vector_component;
    /// Imposed static contact angle in radians
    const number contact_angle;
  };

  /**
   * @brief Simulation of wetting at the bottom wall. This simulates the modelling problem described
   * in "A conservative level set method for contact line dynamics" by Zahedi et al. (2009).
   * DOI: https://doi.org/10.1016/j.jcp.2009.05.043
   *
   * @tparam dim Number of spatial dimensions of the simulation
   * @tparam number Numeric type used for computations (e.g., float, double)
   */
  template <int dim, typename number>
  class SimulationWallWetting : public LevelSet::ReinitializationCase<dim, number>
  {
  public:
    /**
     * @brief Constructor of the wall wetting simulation using wetting boundary condition
     *
     * @param[in] parameter_file Parameter file associated to the simulation.
     *
     * @param[in] mpi_communicator MPI communicator.
     */
    SimulationWallWetting(std::string parameter_file, const MPI_Comm mpi_communicator)
      : LevelSet::ReinitializationCase<dim, number>(parameter_file, mpi_communicator)
    {}

    /**
     * @brief Add parameters that are specific to the simulation.
     *
     * @param[in, out] prm ParameterHandler object to which parameters can be added
     *
     * @return Boolean indicating if parameters should be printed
     */
    bool
    add_simulation_specific_parameters(dealii::ParameterHandler &prm) final
    {
      prm.enter_subsection("simulation specific");
      {
        prm.add_parameter("contact angle",
                          contact_angle_deg,
                          "contact angle at the bottom wall",
                          dealii::Patterns::Double(0, 180));
        prm.add_parameter(
          "gamma factor",
          gamma_factor,
          "factor multiplying epsilon_n in the computation of the normal vector filter parameter as defined by Zahedi et al. (2009)",
          dealii::Patterns::Double());
        prm.add_parameter(
          "time-step factor",
          time_step_factor,
          "time-step scaling factor; it is used to conduct a time-step sensitivity analysis",
          dealii::Patterns::Double());
        prm.add_parameter(
          "output contact angle evolution",
          output_contact_angle_evolution,
          "if set to 'true', it outputs the contact angle evolution in a .txt file in the output directory",
          dealii::Patterns::Bool());
      }
      prm.leave_subsection();
      return this->parameters.base.do_print_parameters;
    }

    /**
     * @brief Create the spatial discretization of the problem.
     */
    void
    create_spatial_discretization() final
    {
      // Check that the simulation is run in 2D
      AssertDimension(dim, 2);

      // Generate triangulation
      this->triangulation =
        std::make_shared<dealii::parallel::distributed::Triangulation<dim>>(this->mpi_communicator);

      // Create mesh
      std::vector<unsigned>  refinements(2, 25);
      const dealii::Point<2> bottom_left(x_min, y_min);
      const dealii::Point<2> top_right(x_max, y_max);
      dealii::GridGenerator::subdivided_hyper_rectangle(
        *this->triangulation, refinements, bottom_left, top_right, true);
      this->triangulation->refine_global(this->parameters.base.global_refinements);
    }

    /**
     * @brief Set boundary conditions.
     */
    void
    set_boundary_conditions() final
    {
      // Face numbering according to the deal.II colorize flag
      const auto [bottom_bc, upper_bc, left_bc, right_bc, front_bc, back_bc] =
        get_colorized_rectangle_boundary_ids<dim>();

      // Wetting boundary condition
      const number contact_angle_rad =
        MeltPoolDG::numbers::compute_angle_in_radians(contact_angle_deg);


      // Component-wise definition of AffineConstraints objects is required. Therefore, one
      // function per component has to be instantiated.
      this->attach_boundary_condition(
        {bottom_bc,
         std::make_shared<BottomBoundaryNormalPerComponent<dim, number>>(0, contact_angle_rad)},
        "nx",
        "normal_vector");
      this->attach_boundary_condition(
        {bottom_bc,
         std::make_shared<BottomBoundaryNormalPerComponent<dim, number>>(1, contact_angle_rad)},
        "ny",
        "normal_vector");
    }

    /**
     * @brief Set initial level-set field condition.
     */
    void
    set_field_conditions() final
    {
      // Compute normal diffusion factor (we consider a static mesh)
      cell_size_min =
        dealii::GridTools::minimal_cell_diameter(*this->triangulation) / std::sqrt(dim);
      epsilon = this->parameters.reinit.compute_interface_thickness_parameter_epsilon(
        cell_size_min / this->parameters.reinit.fe.get_n_subdivisions());

      // Compute time-step and change data value
      /* dt = h^2 / [2 * (epsilon_n + epsilon_t)] * time_step_factor */
      number time_step =
        time_step_factor * 0.5 * dealii::Utilities::fixed_power<2>(cell_size_min) /
        (epsilon * (1 + this->parameters.reinit.hyperbolic.cg.tangential_diffusion_factor));
      this->parameters.time_stepping.time_step_size = time_step;

      // Compute normal vector filtering factor and change data value
      gamma = dealii::Utilities::fixed_power<2>(
        gamma_factor * this->parameters.reinit.interface_thickness_parameter.value);
      this->parameters.normal_vec.filter_parameter = gamma;

      // Initial level-set condition
      this->attach_initial_condition(std::make_shared<InitialLevelSet<dim, number>>(x_interface,
                                                                                    epsilon),
                                     "level_set");
    }

    /**
     * @brief Post-process the solution to extract the contact angle at the boundary for the current
     * time-step and write it in an output file.
     *
     * @param[in] generic_data_out GenericDataOut object containing solution information
     */
    void
    do_postprocessing(
      [[maybe_unused]] const GenericDataOut<dim, double> &generic_data_out) const final
    {
      // Do nothing if output is not requested
      if (not this->parameters.output.do_user_defined_postprocessing)
        return;

      // Compute contact angle
      number contact_angle;
      compute_contact_angle_at_boundary(contact_angle, generic_data_out);

      if (dealii::Utilities::MPI::this_mpi_process(this->mpi_communicator) == 0)
        {
          std::cout << "| Static contact angle: " << this->contact_angle_deg << std::endl;
          std::cout << "| Computed contact angle: " << contact_angle << std::endl;

          if (output_contact_angle_evolution)
            {
              // Add values to table
              postprocess_table.add_value("time", generic_data_out.get_time());
              postprocess_table.set_scientific("time", true);
              postprocess_table.set_scientific("time", 5);
              postprocess_table.add_value("contact_angle", contact_angle);

              // Write in output file
              namespace fs = std::filesystem;
              std::ofstream output(fs::path(this->parameters.output.directory) /
                                   fs::path(this->parameters.output.paraview.filename + ".txt"));
              postprocess_table.write_text(output);
            }
        } // End rank 0 scope
    }

    /**
     * @brief Computes the position-weighted contact angle at the bottom boundary.
     *
     * This function evaluates the contact angle between the fluid interface and the bottom wall of
     * the domain. It traverses the interface using `LevelSet::Tools::evaluate_at_interface`,
     * identifies points near the bottom boundary, and computes the contact angle
     * using a scalar product between the wall normal and the gradient of the level set indicator.
     *
     * The resulting contact angle is weighted based on the vertical distance from the
     * interface point to the bottom wall, so that points closer to the wall have a
     * stronger influence on the result.
     *
     * @param[out] contact_angle Computed contact angle value in degrees
     *
     * @param[in] generic_data_out GenericDataOut object containing solution information
     */
    void
    compute_contact_angle_at_boundary(number                            &contact_angle,
                                      const GenericDataOut<dim, double> &generic_data_out) const
    {
      const auto &dof_handler      = generic_data_out.get_dof_handler("psi");
      const auto &mapping          = generic_data_out.get_mapping();
      const auto &level_set_vector = generic_data_out.get_vector("psi");

      number contact_angle_sum = 0.0;
      number total_weight      = 0.0;

      const unsigned int n_subdivisions = 2;
      const number       tolerance      = 3 * epsilon;
      const number       contour_value  = 0.0;

      // Bottom wall normal
      dealii::Tensor<1, dim> wall_normal;
      wall_normal[1] = 1.0; // unit normal in +y direction

      // Tolerance identifying value at the bottom wall
      const number tolerance_bottom_wall = 0.25 * cell_size_min;

      // Evaluate the contact angle
      LevelSet::Tools::evaluate_at_interface<dim, number>(
        dof_handler,
        mapping,
        level_set_vector,
        [&](const auto &cell,
            const auto &interface_points,
            const auto &reference_points,
            const auto & /*JxW*/) {
          const auto           &fe = dof_handler.get_fe();
          dealii::FEValues<dim> fe_values(mapping,
                                          fe,
                                          dealii::Quadrature<dim>(reference_points),
                                          dealii::update_gradients);
          fe_values.reinit(cell);
          std::vector<dealii::Tensor<1, dim>> grad_phi(reference_points.size());
          fe_values.get_function_gradients(level_set_vector, grad_phi);

          for (unsigned int i = 0; i < reference_points.size(); ++i)
            {
              // Extract real points and level-set indicator gradient values
              const auto &point_i    = interface_points[i];
              const auto &grad_phi_i = grad_phi[i];

              // Check if point is at the bottom wall
              const number distance_to_wall = std::abs(point_i[dim - 1] - y_min);
              if (distance_to_wall < tolerance_bottom_wall)
                {
                  // Compute contact angle with normal vector and grad_phi scalar product
                  const number cos_alpha = grad_phi_i * wall_normal / (grad_phi_i.norm() + 1e-16);
                  const number alpha_rad = std::acos(cos_alpha);

                  // Sum weighted contributions, so that if multiple points are considered, the
                  // closest to the bottom wall have more weight
                  const number weight = (tolerance - distance_to_wall) / tolerance;
                  contact_angle_sum += weight * alpha_rad * 180.0 / dealii::numbers::PI;
                  total_weight += weight;
                }
            }
        },
        contour_value,
        n_subdivisions,
        tolerance,
        true /*use_mca=*/);

      // Sum results from all processes
      const number global_contact_angle_sum =
        dealii::Utilities::MPI::sum(contact_angle_sum, MPI_COMM_WORLD);
      const number global_total_weight = dealii::Utilities::MPI::sum(total_weight, MPI_COMM_WORLD);
      contact_angle = (global_total_weight > 0) ? global_contact_angle_sum / global_total_weight :
                                                  std::numeric_limits<number>::quiet_NaN();
    }

  private:
    /* Domain dimensions */
    /// Minimal x value of the domain
    number x_min = 0.0;
    /// Maximal x value of the domain
    number x_max = 2.0;
    /// Minimal y value of the domain
    number y_min = x_min;
    /// Maximal y value of the domain
    number y_max = x_max;

    /// Interface location
    number x_interface = 1.0;

    /// Contact angle on bottom wall
    number contact_angle_deg = 45.0;

    /// Factor multiplying the normal diffusion coefficient in the computation of the normal vector
    /// filter parameter as defined in Zahedi et al. (2009)
    number gamma_factor = 1.0;

    /// Normal vector filter parameter
    number gamma;

    /// Interface thickness parameter
    number epsilon;

    /// Minimum cell size
    number cell_size_min;

    /// Time-step scaling factor (for sensitivity analysis on the time-step)
    number time_step_factor = 1.0;

    /// Contact angle post-processing table
    mutable dealii::TableHandler postprocess_table;

    /// Bool output contact-angle evolution in a .txt file
    bool output_contact_angle_evolution = true;
  };

} // namespace MeltPoolDG::Simulation::WallWetting
