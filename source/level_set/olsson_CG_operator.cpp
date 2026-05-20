#include <deal.II/matrix_free/tools.h>

#include <meltpooldg/core/exceptions.hpp>
#include <meltpooldg/level_set/olsson_CG_operator.hpp>
#include <meltpooldg/utilities/dealii_tensor.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>

#include <cmath>


namespace MeltPoolDG::LevelSet
{
  using namespace dealii;

  /**
   * @brief Computes the compressive flux term in the reinitialization equation for a level set formulation.
   *
   * @tparam number A scalar numeric type (e.g., float, double).
   *
   * @param[in] psi The level set value at a point.
   *
   * @return The value of the compressive flux evaluated at the given \p psi.
   */
  template <typename number>
  number
  compressive_flux(const number psi)
  {
    return number(0.5) * (number(1.) - psi * psi);
  }

  template <int dim, typename number>
  OlssonOperator<dim, number>::OlssonOperator(const ScratchData<dim, dim, number> &scratch_data_in,
                                              const ReinitializationData<number>  &reinit_data_in,
                                              const int              ls_n_subdivisions,
                                              const BlockVectorType &n_in,
                                              const unsigned int     reinit_dof_idx_in,
                                              const unsigned int     reinit_quad_idx_in,
                                              const unsigned int     ls_dof_idx_in,
                                              const unsigned int     normal_dof_idx_in)
    : scratch_data(scratch_data_in)
    , reinit_data(reinit_data_in)
    , ls_n_subdivisions(ls_n_subdivisions)
    , normal_vec(n_in)
    , reinit_quad_idx(reinit_quad_idx_in)
    , normal_dof_idx(normal_dof_idx_in)
    , ls_dof_idx(ls_dof_idx_in)
    , tolerance_normal_vector(UtilityFunctions::compute_numerical_zero_of_norm<dim, number>(
        scratch_data.get_triangulation(),
        scratch_data.get_mapping()))
  {
    this->reset_dof_index(reinit_dof_idx_in);
  }

  template <int dim, typename number>
  void
  OlssonOperator<dim, number>::set_wetting_boundary_condition_ids(
    std::vector<dealii::types::boundary_id> &&wetting_bc_ids_in)
  {
    wetting_bc_ids                = std::move(wetting_bc_ids_in);
    enable_boundary_face_integral = (wetting_bc_ids.size() > 0);
  }

  template <int dim, typename number>
  void
  OlssonOperator<dim, number>::compute_system_matrix_and_rhs(const VectorType &levelset_old,
                                                             VectorType       &rhs) const
  {
    AssertThrowZeroTimeIncrement(this->time_increment);

    FEValues<dim>      fe_values(scratch_data.get_mapping(),
                            scratch_data.get_dof_handler(this->dof_idx).get_fe(),
                            scratch_data.get_quadrature(reinit_quad_idx),
                            update_values | update_gradients | update_quadrature_points |
                              update_JxW_values);
    const unsigned int dofs_per_cell = scratch_data.get_n_dofs_per_cell(this->dof_idx);

    FullMatrix<number> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<number>     cell_rhs(dofs_per_cell);

    const unsigned int n_q_points = fe_values.get_quadrature().size();

    std::vector<number>                 psi_at_q(n_q_points);
    std::vector<Tensor<1, dim, number>> grad_psi_at_q(n_q_points, Tensor<1, dim, number>());
    std::vector<Tensor<1, dim, number>> normal_at_q(n_q_points, Tensor<1, dim, number>());

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    rhs                 = 0.0;
    this->system_matrix = 0.0;

    for (const auto &cell : scratch_data.get_dof_handler(this->dof_idx).active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell_matrix = 0.0;
          cell_rhs    = 0.0;
          fe_values.reinit(cell);

          const number epsilon_cell = reinit_data.compute_interface_thickness_parameter_epsilon(
            cell->diameter() / std::sqrt(dim) / ls_n_subdivisions);

          Assert(
            epsilon_cell > 0.0,
            ExcMessage(
              "Reinitialization: the value of epsilon for the reinitialization function must be larger than zero!"));

          fe_values.get_function_values(levelset_old,
                                        psi_at_q); // compute values of old solution at tau_n
          fe_values.get_function_gradients(
            levelset_old, grad_psi_at_q); // compute gradients of old solution at tau_n
          NormalVectorOperator<dim, number>::get_unit_normals_at_quadrature(
            fe_values, this->normal_vec, normal_at_q, tolerance_normal_vector);

          for (const unsigned int q_index : fe_values.quadrature_point_indices())
            {
              for (const unsigned int i : fe_values.dof_indices())
                {
                  const number nTimesGradient_i =
                    normal_at_q[q_index] * fe_values.shape_grad(i, q_index);

                  for (const unsigned int j : fe_values.dof_indices())
                    {
                      const number nTimesGradient_j =
                        normal_at_q[q_index] * fe_values.shape_grad(j, q_index);
                      //clang-format off
                      cell_matrix(i, j) +=
                        (fe_values.shape_value(i, q_index) * fe_values.shape_value(j, q_index) +
                         this->time_increment * epsilon_cell * nTimesGradient_i *
                           nTimesGradient_j) *
                        fe_values.JxW(q_index);
                      //clang-format on
                    }

                  const number diffRhs =
                    epsilon_cell * normal_at_q[q_index] * grad_psi_at_q[q_index];

                  cell_rhs(i) += (compressive_flux(psi_at_q[q_index]) - diffRhs) *
                                 nTimesGradient_i * this->time_increment * fe_values.JxW(q_index);
                }
            } // end loop over gauss points
          // assembly
          cell->get_dof_indices(local_dof_indices);

          scratch_data.get_constraint(this->dof_idx)
            .distribute_local_to_global(
              cell_matrix, cell_rhs, local_dof_indices, this->system_matrix, rhs);
        }

    this->system_matrix.compress(VectorOperation::add);
    rhs.compress(VectorOperation::add);
  }

  template <int dim, typename number>
  void
  OlssonOperator<dim, number>::vmult(VectorType &dst, const VectorType &src) const
  {
    scratch_data.get_matrix_free().template loop<VectorType, VectorType>(
      [&](const auto &matrix_free, auto &dst, const auto &src, auto cell_range) {
        FECellIntegrator<dim, 1, number> delta_psi(matrix_free, this->dof_idx, reinit_quad_idx);
        for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
          {
            delta_psi.reinit(cell);
            delta_psi.read_dof_values(src);

            tangent_cell_operation(delta_psi);

            delta_psi.distribute_local_to_global(dst);
          }
      }, // cell loop
      [&](const auto &,
          auto &,
          const auto &,
          auto /*face_range*/) { /*do nothing*/ }, // internal face loop
      [&](const auto &matrix_free, auto &dst, const auto &src, auto face_range) {
        if (enable_boundary_face_integral)
          tangent_boundary_loop(matrix_free, dst, src, face_range);
      },
      dst,
      src,
      true /*zero dst vector*/);
  }

  template <int dim, typename number>
  void
  OlssonOperator<dim, number>::create_rhs(VectorType &dst, const VectorType &src) const
  {
    AssertThrowZeroTimeIncrement(this->time_increment);

    scratch_data.get_matrix_free().template loop<VectorType, VectorType>(
      [&](const auto &matrix_free, auto &dst, const auto &src, auto macro_cells) {
        FECellIntegrator<dim, 1, number>   rhs(matrix_free, this->dof_idx, reinit_quad_idx);
        FECellIntegrator<dim, 1, number>   psi_old(matrix_free, ls_dof_idx, reinit_quad_idx);
        FECellIntegrator<dim, dim, number> normal_vector(scratch_data.get_matrix_free(),
                                                         normal_dof_idx,
                                                         reinit_quad_idx);

        for (unsigned int cell = macro_cells.first; cell < macro_cells.second; ++cell)
          {
            rhs.reinit(cell);

            psi_old.reinit(cell);
            psi_old.read_dof_values_plain(src);
            psi_old.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

            normal_vector.reinit(cell);
            normal_vector.read_dof_values_plain(this->normal_vec);
            normal_vector.evaluate(EvaluationFlags::values);

            for (unsigned int q_index = 0; q_index < rhs.n_q_points; ++q_index)
              {
                const scalar val = psi_old.get_value(q_index);

                // Normal unit vector
                const vector unit_normal_interface =
                  normalize<dim>(normal_vector.get_value(q_index), tolerance_normal_vector);
                unit_normal[cell * rhs.n_q_points + q_index]  = unit_normal_interface;
                solution_old[cell * rhs.n_q_points + q_index] = val;

                const auto normal_diffusive_flux =
                  (scalar_product(psi_old.get_gradient(q_index), unit_normal_interface) *
                   unit_normal_interface);
                vector tangential_diffusive_flux =
                  psi_old.get_gradient(q_index) - normal_diffusive_flux;

                rhs.submit_gradient(compressive_flux(val) * unit_normal_interface
                                      // Normal contribution
                                      - normal_diffusion_length[cell] * normal_diffusive_flux
                                      // Tangential contribution
                                      -
                                      tangential_diffusion_length[cell] * tangential_diffusive_flux,
                                    q_index);
              }

            rhs.integrate_scatter(EvaluationFlags::gradients, dst);
          }
      },
      [&](const auto &,
          auto &,
          const auto &,
          auto /*face_range*/) { /*do nothing*/ }, // internal face loop
      [&](const auto &matrix_free, auto &dst, const auto &src, auto face_range) {
        // only for wetting boundary condition
        if (enable_boundary_face_integral)
          rhs_boundary_loop(matrix_free, dst, src, face_range);
      },
      dst,
      src,
      true /*zero out dst*/);
  }

  template <int dim, typename number>
  void
  OlssonOperator<dim, number>::compute_system_matrix_from_matrixfree(
    TrilinosWrappers::SparseMatrix &system_matrix) const
  {
    system_matrix = 0.0;

    // note: not thread safe!!!
    const auto                        &matrix_free = scratch_data.get_matrix_free();
    FEFaceIntegrator<dim, dim, number> normal_face_eval(matrix_free,
                                                        true /*is_interior_face*/,
                                                        normal_dof_idx,
                                                        reinit_quad_idx);

    unsigned int old_cell_index = dealii::numbers::invalid_unsigned_int;


    // Compute diagonal of the operator using MatrixFreeTools.
    //
    // We distinguish two cases:
    // 1. If boundary face integrals are enabled (`enable_boundary_face_integral == true`),
    //    we must pass a valid boundary operation to `compute_diagonal`.
    // 2. If boundary face integrals are disabled, we use a simplified variant
    //    without boundary or interior face operations.
    //
    // Note: The boundary face evaluation used in 2. is only possible when the matrix-free
    //       framework has face loops enabled. If face loops are disabled and we attempt to use
    //       boundary face operations, this will lead to an assertion failure in deal.II:
    //
    //       FEEvaluationData(..., is_face = true): !data->data.empty()
    if (enable_boundary_face_integral)
      {
        MatrixFreeTools::template compute_matrix<dim, -1, 0, 1, number, VectorizedArray<number>>(
          matrix_free,
          scratch_data.get_constraint(this->dof_idx),
          system_matrix,
          [&](auto &delta_psi) {
            const unsigned int current_cell_index = delta_psi.get_current_cell_index();

            tangent_cell_operation(delta_psi);

            old_cell_index = current_cell_index;
          },
          // face operation
          {},
          // boundary operation
          [&](auto &face_eval) { tangent_boundary_face_operation(face_eval, normal_face_eval); },
          this->dof_idx,
          reinit_quad_idx);
      }
    else
      {
        // Simpler version without boundary face operations
        MatrixFreeTools::template compute_matrix<dim, -1, 0, 1, number, VectorizedArray<number>>(
          matrix_free,
          scratch_data.get_constraint(this->dof_idx),
          system_matrix,
          [&](auto &delta_psi) {
            const unsigned int current_cell_index = delta_psi.get_current_cell_index();

            tangent_cell_operation(delta_psi);

            old_cell_index = current_cell_index;
          },
          this->dof_idx,
          reinit_quad_idx);
      }
  }

  template <int dim, typename number>
  void
  OlssonOperator<dim, number>::compute_inverse_diagonal_from_matrixfree(VectorType &diagonal) const
  {
    scratch_data.initialize_dof_vector(diagonal, this->dof_idx);

    // note: not thread safe!!!
    const auto &matrix_free = scratch_data.get_matrix_free();

    unsigned int old_cell_index = dealii::numbers::invalid_unsigned_int;

    FEFaceIntegrator<dim, dim, number> normal_face_eval(matrix_free,
                                                        true /*is_interior_face*/,
                                                        normal_dof_idx,
                                                        reinit_quad_idx);

    // Compute diagonal of the operator using MatrixFreeTools.
    //
    // We distinguish two cases:
    // 1. If boundary face integrals are enabled (`enable_boundary_face_integral == true`),
    //    we must pass a valid boundary operation to `compute_diagonal`.
    // 2. If boundary face integrals are disabled, we use a simplified variant
    //    without boundary or interior face operations.
    //
    // Note: The boundary face evaluation used in 2. is only possible when the matrix-free
    //       framework has face loops enabled. If face loops are disabled and we attempt to use
    //       boundary face operations, this will lead to an assertion failure in deal.II:
    //
    //       FEEvaluationData(..., is_face = true): !data->data.empty()
    if (enable_boundary_face_integral)
      {
        // Full version with boundary face operations
        MatrixFreeTools::template compute_diagonal<dim, -1, 0, 1, number, VectorizedArray<number>>(
          matrix_free,
          diagonal,
          [&](auto &delta_psi) {
            const unsigned int current_cell_index = delta_psi.get_current_cell_index();

            tangent_cell_operation(delta_psi);

            old_cell_index = current_cell_index;
          },
          // face operation
          {},
          // boundary operation
          [&](auto &face_eval) { tangent_boundary_face_operation(face_eval, normal_face_eval); },
          this->dof_idx,
          reinit_quad_idx);
      }
    else
      {
        // Simpler version without boundary face operations
        MatrixFreeTools::template compute_diagonal<dim, -1, 0, 1, number, VectorizedArray<number>>(
          matrix_free,
          diagonal,
          [&](auto &delta_psi) {
            const unsigned int current_cell_index = delta_psi.get_current_cell_index();

            tangent_cell_operation(delta_psi);

            old_cell_index = current_cell_index;
          },
          this->dof_idx,
          reinit_quad_idx);
      }

    // ... and invert it
    const number linfty_norm = std::max(1.0, diagonal.linfty_norm());
    for (auto &i : diagonal)
      i = (std::abs(i) > 1.0e-14 * linfty_norm) ? (1.0 / i) : 1.0;
  }


  template <int dim, typename number>
  void
  OlssonOperator<dim, number>::tangent_cell_operation(
    FECellIntegrator<dim, 1, number> &delta_psi) const
  {
    delta_psi.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

    for (unsigned int q_index = 0; q_index < delta_psi.n_q_points; q_index++)
      {
        const auto &cell                  = delta_psi.get_current_cell_index();
        const auto  unit_normal_interface = unit_normal[cell * delta_psi.n_q_points + q_index];

        const vector normal_diffusive_flux =
          scalar_product(delta_psi.get_gradient(q_index), unit_normal_interface) *
          unit_normal_interface;

        const vector tangential_diffusive_flux =
          delta_psi.get_gradient(q_index) - normal_diffusive_flux;

        delta_psi.submit_value(this->time_increment_inv * delta_psi.get_value(q_index), q_index);
        delta_psi.submit_gradient(
          // Normal contribution
          (normal_diffusion_length[cell] * normal_diffusive_flux
           // Tangential contribution
           + tangential_diffusion_length[cell] * tangential_diffusive_flux),
          q_index);
      }

    delta_psi.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
  }

  template <int dim, typename number>
  void
  OlssonOperator<dim, number>::tangent_boundary_loop(
    const MatrixFree<dim, number>        &matrix_free,
    VectorType                           &dst,
    const VectorType                     &src,
    std::pair<unsigned int, unsigned int> face_range) const
  {
    FEFaceIntegrator<dim, 1, number>   delta_psi_face_eval(matrix_free,
                                                         true /*is_interior_face*/,
                                                         this->dof_idx,
                                                         reinit_quad_idx);
    FEFaceIntegrator<dim, dim, number> normal_face_eval(matrix_free,
                                                        true /*is_interior_face*/,
                                                        normal_dof_idx,
                                                        reinit_quad_idx);

    for (unsigned int face = face_range.first; face < face_range.second; ++face)
      {
        delta_psi_face_eval.reinit(face);

        // check whether the face has a prescribed wetting angle
        const types::boundary_id bc_index = scratch_data.get_matrix_free().get_boundary_id(
          delta_psi_face_eval.get_current_cell_index());

        if (std::find(wetting_bc_ids.begin(), wetting_bc_ids.end(), bc_index) ==
            wetting_bc_ids.end())
          continue;


        delta_psi_face_eval.read_dof_values(src);

        tangent_boundary_face_operation(delta_psi_face_eval, normal_face_eval);

        delta_psi_face_eval.distribute_local_to_global(dst);
      }
  }

  template <int dim, typename number>
  void
  OlssonOperator<dim, number>::rhs_boundary_loop(
    const MatrixFree<dim, number>        &matrix_free,
    VectorType                           &dst,
    const VectorType                     &src,
    std::pair<unsigned int, unsigned int> face_range) const
  {
    FEFaceIntegrator<dim, 1, number>   face_eval(matrix_free,
                                               true /*is_interior_face*/,
                                               ls_dof_idx,
                                               reinit_quad_idx);
    FEFaceIntegrator<dim, dim, number> normal_face_eval(matrix_free,
                                                        true /*is_interior_face*/,
                                                        normal_dof_idx,
                                                        reinit_quad_idx);

    for (unsigned int face = face_range.first; face < face_range.second; ++face)
      {
        face_eval.reinit(face);

        // check whether the face has a prescribed wetting angle
        const types::boundary_id bc_index =
          scratch_data.get_matrix_free().get_boundary_id(face_eval.get_current_cell_index());

        if (std::find(wetting_bc_ids.begin(), wetting_bc_ids.end(), bc_index) ==
            wetting_bc_ids.end())
          continue;

        face_eval.read_dof_values_plain(src);

        rhs_boundary_face_operation(face_eval, normal_face_eval);

        face_eval.distribute_local_to_global(dst);
      }
  }

  template <int dim, typename number>
  void
  OlssonOperator<dim, number>::tangent_boundary_face_operation(
    FEFaceIntegrator<dim, 1, number>   &face_eval,
    FEFaceIntegrator<dim, dim, number> &normal_face_eval) const
  {
    face_eval.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

    const auto &cell = face_eval.get_current_cell_index();
    normal_face_eval.reinit(cell);
    normal_face_eval.read_dof_values_plain(normal_vec);
    normal_face_eval.evaluate(EvaluationFlags::values);

    // normal diffusion coefficient
    const auto eps_n = normal_diffusion_length[cell];

    // tangential diffusion coefficient
    const auto eps_t = tangential_diffusion_length[cell];

    for (unsigned int q = 0; q < face_eval.n_q_points; ++q)
      {
        // interface normal vector
        const vector unit_normal_interface =
          normalize<dim>(normal_face_eval.get_value(q), tolerance_normal_vector);

        // Decompose gradients into normal ...
        const vector normal_diffusive_flux =
          (scalar_product(face_eval.get_gradient(q), unit_normal_interface) *
           VectorTools::to_vector<dim>(unit_normal_interface));

        // ... and tangential components
        const vector tangential_diffusive_flux = face_eval.get_gradient(q) - normal_diffusive_flux;

        // normal diffusion
        auto flux = eps_n * normal_diffusive_flux;

        // Tangential contribution
        flux += eps_t * tangential_diffusive_flux;

        // NOTE: Here, the sign is the opposite of what was developed by hand
        face_eval.submit_value(scalar_product(flux, face_eval.normal_vector(q)), q);
      }
    face_eval.integrate(EvaluationFlags::values);
  }

  template <int dim, typename number>
  void
  OlssonOperator<dim, number>::rhs_boundary_face_operation(
    FEFaceIntegrator<dim, 1, number>   &face_eval, // old psi,
    FEFaceIntegrator<dim, dim, number> &normal_face_eval) const
  {
    face_eval.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

    const auto &cell = face_eval.get_current_cell_index();
    normal_face_eval.reinit(cell);
    normal_face_eval.read_dof_values_plain(normal_vec);
    normal_face_eval.evaluate(EvaluationFlags::values);

    // normal diffusion coefficient
    const auto eps_n = normal_diffusion_length[cell];

    // tangential diffusion coefficient
    const auto eps_t = tangential_diffusion_length[cell];

    for (unsigned int q = 0; q < face_eval.n_q_points; ++q)
      {
        // interface normal vector
        const vector unit_normal_interface =
          normalize<dim>(normal_face_eval.get_value(q), tolerance_normal_vector);

        // Decompose gradients into normal ...
        const vector normal_diffusive_flux =
          (scalar_product(face_eval.get_gradient(q), unit_normal_interface) *
           VectorTools::to_vector<dim>(unit_normal_interface));

        // ... and tangential components
        const vector tangential_diffusive_flux = face_eval.get_gradient(q) - normal_diffusive_flux;

        // compressive flux
        auto flux = -compressive_flux(face_eval.get_value(q)) * unit_normal_interface;

        // normal diffusion
        flux += eps_n * normal_diffusive_flux;

        // tangential diffusion
        flux += eps_t * tangential_diffusive_flux;

        // NOTE: Here, the sign is the opposite of what was developed by hand
        face_eval.submit_value(-scalar_product(flux, face_eval.normal_vector(q)), q);
      }
    face_eval.integrate(EvaluationFlags::values);
  }

  template <int dim, typename number>
  void
  OlssonOperator<dim, number>::reinit()
  {
    if (reinit_data.linear_solver.do_matrix_free)
      {
        normal_diffusion_length.resize_fast(scratch_data.get_matrix_free().n_cell_batches());
        tangential_diffusion_length.resize_fast(scratch_data.get_matrix_free().n_cell_batches());

        unit_normal.resize_fast(scratch_data.get_matrix_free().n_cell_batches() *
                                scratch_data.get_n_q_points(reinit_quad_idx));
        solution_old.resize_fast(scratch_data.get_matrix_free().n_cell_batches() *
                                 scratch_data.get_n_q_points(reinit_quad_idx));

        for (unsigned int cell = 0; cell < scratch_data.get_matrix_free().n_cell_batches(); ++cell)
          {
            normal_diffusion_length[cell] =
              reinit_data.compute_interface_thickness_parameter_epsilon(
                scratch_data.get_cell_sizes()[cell] / ls_n_subdivisions);
            tangential_diffusion_length[cell] =
              normal_diffusion_length[cell] * reinit_data.hyperbolic.cg.tangential_diffusion_factor;
          }
      }
    else
      {
        this->reinit_sparsity_pattern(scratch_data);
      }
  }

  template class OlssonOperator<1, double>;
  template class OlssonOperator<2, double>;
  template class OlssonOperator<3, double>;
} // namespace MeltPoolDG::LevelSet
