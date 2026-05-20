#pragma once

#include <deal.II/base/function.h>

#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/level_set/reinitialization_data.hpp>
#include <meltpooldg/level_set/reinitialization_operation_base.hpp>
#include <meltpooldg/level_set/signed_distance_solver.hpp>
#include <meltpooldg/post_processing/generic_data_out.hpp>
#include <meltpooldg/utilities/journal.hpp>

namespace MeltPoolDG::LevelSet
{
  using namespace dealii;

  /**
   * @brief Geometric level-set reinitialization operation.
   *
   * This class computes a signed-distance representation of a level-set field
   * using a geometric signed-distance solver. Optionally, the resulting signed
   * distance field can be transformed into a smoothed hyperbolic tangent
   * level-set representation.
   *
   * The operation uses the background mesh and finite element space associated
   * with the provided degree-of-freedom index.
   *
   * @tparam dim Spatial dimension.
   * @tparam number Scalar number type.
   */
  template <int dim, typename number>
  class ReinitializationGeometricOperation : public ReinitializationOperationBase<dim, number>
  {
  public:
    /// Distributed vector type used for scalar fields.
    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

    /// Distributed block vector type.
    using BlockVectorType = dealii::LinearAlgebra::distributed::BlockVector<number>;

    /**
     * @brief Constructor.
     *
     * Initializes the geometric reinitialization operation and creates the
     * internal signed-distance solver.
     *
     * @param scratch_data_in Shared scratch data and FE infrastructure.
     * @param reinit_data_in Reinitialization configuration parameters.
     * @param ls_dof_idx_in Degree-of-freedom index of the level-set field.
     * @param ls_quad_idx_in Quadrature index associated with the level-set field.
     */
    ReinitializationGeometricOperation(const ScratchData<dim, dim, number> &scratch_data_in,
                                       const ReinitializationData<number>  &reinit_data_in,
                                       const unsigned int                   ls_dof_idx_in,
                                       const unsigned int                   ls_quad_idx_in);

    /**
     * @brief Initialize the reinitialization operation.
     *
     * Allocates vectors and initializes the signed-distance solver degrees of
     * freedom.
     */
    void
    reinit() override;

    /**
     * @brief Execute the geometric reinitialization procedure.
     *
     * Computes the signed-distance field from the current level-set field and,
     * if enabled, transforms it into a hyperbolic tangent level-set field.
     */
    void
    solve() override;

    /**
     * @brief Set the initial level-set field from a vector.
     *
     * @param solution_level_set_in Input level-set vector.
     */
    void
    set_initial_condition(const VectorType &solution_level_set_in) override;

    /**
     * @brief Set the initial level-set field from an analytical function.
     *
     * The function is interpolated onto the level-set finite element space.
     *
     * @param initial_field_function Analytical initial field function.
     */
    void
    set_initial_condition(const Function<dim> &initial_field_function) override;

    /**
     * @brief Get a constant reference to the level-set field.
     *
     * @return Constant reference to the level-set vector.
     */
    const VectorType &
    get_level_set() const override;

    /**
     * @brief Get a mutable reference to the level-set field.
     *
     * @return Mutable reference to the level-set vector.
     */
    VectorType &
    get_level_set() override;

    /**
     * @brief Attach internal vectors to an external vector list.
     *
     * @param vectors Container receiving pointers to internal vectors.
     */
    void
    attach_vectors([[maybe_unused]] std::vector<VectorType *> &vectors) override;

    /**
     * @brief Attach output vectors for postprocessing and visualization.
     *
     * Adds the signed-distance field and level-set field to the output object.
     *
     * @param data_out Output handler.
     */
    void
    attach_output_vectors(GenericDataOut<dim, number> &data_out) const override;

  private:
    /**
     * @brief Create and configure the signed-distance solver.
     */
    void
    create_solver();

    /**
     * @brief Transform the signed-distance field into a tanh level-set field.
     *
     * The transformation uses a cell-dependent interface thickness parameter
     * computed from the mesh resolution and finite element subdivision data.
     */
    void
    transform_signed_distance_to_tanh_level_set();

  private:
    /// Shared scratch data and finite element infrastructure.
    const ScratchData<dim, dim, number> &scratch_data;

    /// Reinitialization configuration data.
    const ReinitializationData<number> &reinit_data;

    /// Degree-of-freedom index for the level-set field.
    unsigned int ls_dof_idx;

    /// Quadrature index for the level-set field.
    unsigned int ls_quad_idx;

    /// Reinitialized level-set field.
    VectorType level_set;

    /// Geometric signed-distance solver.
    std::unique_ptr<SignedDistanceSolver<dim>> signed_distance_solver;
  };

} // namespace MeltPoolDG::LevelSet
