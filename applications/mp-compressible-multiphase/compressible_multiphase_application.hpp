#pragma once

#include <meltpooldg/compressible_flow/multiphase_level_set_advection.hpp>
#include <meltpooldg/compressible_flow/multiphase_operation.hpp>
#include <meltpooldg/core/simulation_base.hpp>
#include <meltpooldg/post_processing/postprocessor.hpp>
#include <meltpooldg/time_integration/time_iterator.hpp>
#include <meltpooldg/utilities/profiling_monitor.hpp>

#include "compressible_multiphase_case.hpp"

namespace MeltPoolDG::Multiphase
{
  /**
   * @brief Application for the simulation of the compressible multiphase Navier-Stokes equations
   * using cutDG.
   */
  template <int dim, typename number>
  class CompressibleMultiphaseApplication
  {
  public:
    using CaseType   = CompressibleMultiphaseCase<dim, number>;
    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

    /**
     * @brief Constructor.
     *
     * @param simulation_case Pointer to the considered simulation case.
     */
    explicit CompressibleMultiphaseApplication(std::unique_ptr<CaseType> simulation_case)
      : simulation_case(std::move(simulation_case))
    {}

    /**
     * @brief Run the simulation.
     *
     * This function internally sets up all required member data and performs the time loop.
     */
    void
    run();

  private:
    /// Pointer to the considered simulation case
    std::shared_ptr<CaseType> simulation_case;

    /// DoFHandler for the solution vector
    dealii::DoFHandler<dim> dof_handler;

    /// DoFHandler for the level-set field
    dealii::DoFHandler<dim> dof_handler_level_set;

    /// Constraints for the solution vector
    dealii::AffineConstraints<number> constraints;

    /// Constraints for the level-set field
    dealii::AffineConstraints<number> constraints_level_set;

    /// Pointer to the scratch data object
    std::shared_ptr<ScratchData<dim, dim, number>> scratch_data;

    /// Pointer to the time iterator
    std::shared_ptr<TimeIntegration::TimeIterator<number>> time_iterator;

    /// Compressible multiphase operation object
    std::unique_ptr<Multiphase::CompressibleMultiphaseOperation<dim, number>>
      comp_multiphase_operation;

    /// Pointer to the profiling object
    std::unique_ptr<Profiling::ProfilingMonitor<number>> profiling_monitor;

    /// Operator for analytical level-set advection
    LevelSetAdvection<dim, number> level_set_advection_operator;

    /// Index for the solution DoFHandler
    unsigned int comp_multiphase_dof_idx{};

    /// Index for the level-set DoFHandler
    unsigned int level_set_dof_idx{};

    /// Quadrature index for the computation of flow quantities
    unsigned int comp_multiphase_quad_idx{};

    /// Pointer to the post processor object
    std::unique_ptr<Postprocessor<dim, number>> post_processor;

    /// Pointer to the level-set field function
    std::shared_ptr<dealii::Function<dim>> level_set_field_function;

    /// Level-set DoF vector
    VectorType level_set;

    /**
     * @brief Set up all dof related data.
     */
    void
    setup_dof_system();

    /**
     * @brief Initializes the relevant member data required for solving the compressible flow
     * problem and prepares the object for starting the time loop.
     *
     * This includes setting the necessary boundary and initial conditions.
     *
     * @note Internally calls the setup_dof_system() function.
     */
    void
    initialize();

    /**
     *  @brief Perform the output of the results.
     *
     *  @param time_step Current time step size.
     *  @param current_time Current time at t^n.
     */
    void
    output_results(const unsigned int time_step, const number &current_time);

    /**
     * @brief Interpolates the values of a (currently) analytically given level-set function to the
     * level-set dof vector.
     */
    void
    interpolate_initial_level_set();

    /**
     * @brief Do level-set advection and reinitialization.
     *
     * @param time_step Current time step size.
     */
    void
    update_level_set(const number &time_step);
  };

} // namespace MeltPoolDG::Multiphase
