#ifdef MPDG_ENABLE_ADAFLO
#  include <meltpooldg/level_set/curvature_operation_adaflo_wrapper.hpp>
//
#  include <meltpooldg/utilities/journal.hpp>
#  include <meltpooldg/utilities/vector_tools.hpp>

#  include <adaflo/level_set_okz_preconditioner.h>
#  include <adaflo/util.h>

namespace MeltPoolDG::LevelSet
{
  using namespace dealii;

  template <int dim, typename number>
  CurvatureOperationAdaflo<dim, number>::CurvatureOperationAdaflo(
    const ScratchData<dim, dim, number> &scratch_data,
    const int                            advec_diff_dof_idx,
    const int                            normal_vec_dof_idx,
    const int                            curv_dof_idx,
    const int                            curv_quad_idx,
    const VectorType                    &advected_field,
    const LevelSetData<number>          &data_in)
    : scratch_data(scratch_data)
    , advected_field(advected_field)
    , normal_vector_data(data_in.normal_vec)
  {
    (void)normal_vec_dof_idx;

    /**
     * set parameters of adaflo
     */
    set_adaflo_parameters(data_in, advec_diff_dof_idx, curv_dof_idx, curv_quad_idx);

    /**
     * initialize the projection matrix
     */
    projection_matrix     = std::make_shared<adaflo::BlockMatrixExtension>();
    ilu_projection_matrix = std::make_shared<adaflo::BlockILUExtension>();
  }

  template <int dim, typename number>
  void
  CurvatureOperationAdaflo<dim, number>::create_operator()
  {
    curvature_operation = std::make_shared<adaflo::LevelSetOKZSolverComputeCurvature<dim>>(
      scratch_data.get_cell_sizes(),
      normal_vector_operation_adaflo->get_solution_normal_vector(),
      scratch_data.get_constraint(curv_adaflo_params.dof_index_curvature),
      scratch_data.get_constraint(
        curv_adaflo_params
          .dof_index_curvature), // @todo -- check adaflo --> hanging node constraints??
      epsilon_used,
      rhs,
      curv_adaflo_params,
      curvature_field,
      advected_field,
      scratch_data.get_matrix_free(),
      preconditioner,
      projection_matrix,
      ilu_projection_matrix);
  }

  template <int dim, typename number>
  void
  CurvatureOperationAdaflo<dim, number>::create_normal_vector_operator()
  {
    normal_vector_operation_adaflo = std::make_shared<NormalVectorOperationAdaflo<dim, number>>(
      scratch_data,
      normal_vector_data,
      curv_adaflo_params.dof_index_ls,
      curv_adaflo_params.dof_index_normal,
      curv_adaflo_params.quad_index,
      advected_field,
      curv_adaflo_params.epsilon);
  }

  template <int dim, typename number>
  void
  CurvatureOperationAdaflo<dim, number>::reinit()
  {
    /**
     *  initialize the dof vectors
     */
    initialize_vectors();

    adaflo::compute_cell_diameters<dim>(scratch_data.get_matrix_free(),
                                        curv_adaflo_params.dof_index_ls,
                                        cell_diameters,
                                        cell_diameter_min,
                                        cell_diameter_max);

    epsilon_used = cell_diameter_max * curv_adaflo_params.epsilon;

    if (!normal_vector_operation_adaflo)
      create_normal_vector_operator();
    if (!curvature_operation)
      create_operator();

    /**
     * initialize the preconditioner -->  @todo: currently not used in adaflo
     */
    initialize_mass_matrix_diagonal<dim, number>(scratch_data.get_matrix_free(),
                                                 scratch_data.get_constraint(
                                                   curv_adaflo_params.dof_index_curvature),
                                                 curv_adaflo_params.dof_index_curvature,
                                                 curv_adaflo_params.quad_index,
                                                 preconditioner);

    initialize_projection_matrix<dim, number, VectorizedArray<number>>(
      scratch_data.get_matrix_free(),
      scratch_data.get_constraint(curv_adaflo_params.dof_index_curvature),
      curv_adaflo_params.dof_index_curvature,
      curv_adaflo_params.quad_index,
      epsilon_used,
      curv_adaflo_params.epsilon,
      cell_diameters,
      *projection_matrix,
      *ilu_projection_matrix);

    normal_vector_operation_adaflo->reinit();
  }

  template <int dim, typename number>
  void
  CurvatureOperationAdaflo<dim, number>::update_normal_vector()
  {
    advected_field.update_ghost_values();
    normal_vector_operation_adaflo->solve();
    advected_field.zero_out_ghost_values();
  }

  template <int dim, typename number>
  void
  CurvatureOperationAdaflo<dim, number>::solve()
  {
    advected_field.update_ghost_values();
    normal_vector_operation_adaflo->solve();
    curvature_operation->compute_curvature(
      false); // @todo: adaflo does not use the boolean function argument

    const int verbosity_l2_norm = dim > 1 ? 1 : 2;
    Journal::print_formatted_norm<number>(
      scratch_data.get_pcout(std::max(normal_vector_data.verbosity_level, verbosity_l2_norm)),
      [&]() -> number {
        return VectorTools::compute_norm<dim, number>(get_curvature(),
                                                      scratch_data,
                                                      curv_adaflo_params.dof_index_curvature,
                                                      curv_adaflo_params.quad_index);
      },
      "curvature",
      "curvature_adaflo",
      10 /*precision*/
    );
    advected_field.zero_out_ghost_values();
  }

  template <int dim, typename number>
  const LinearAlgebra::distributed::Vector<number> &
  CurvatureOperationAdaflo<dim, number>::get_curvature() const
  {
    return curvature_field;
  }

  template <int dim, typename number>
  LinearAlgebra::distributed::Vector<number> &
  CurvatureOperationAdaflo<dim, number>::get_curvature()
  {
    return curvature_field;
  }

  template <int dim, typename number>
  const LinearAlgebra::distributed::BlockVector<number> &
  CurvatureOperationAdaflo<dim, number>::get_normal_vector() const
  {
    return normal_vector_operation_adaflo->get_solution_normal_vector();
  }

  template <int dim, typename number>
  LinearAlgebra::distributed::BlockVector<number> &
  CurvatureOperationAdaflo<dim, number>::get_normal_vector()
  {
    return normal_vector_operation_adaflo->get_solution_normal_vector();
  }

  template <int dim, typename number>
  void
  CurvatureOperationAdaflo<dim, number>::attach_vectors(
    std::vector<LinearAlgebra::distributed::Vector<number> *> &vectors)
  {
    normal_vector_operation_adaflo->attach_vectors(vectors);
    vectors.push_back(&curvature_field);
  }

  template <int dim, typename number>
  void
  CurvatureOperationAdaflo<dim, number>::set_adaflo_parameters(
    const LevelSetData<number> &parameters,
    const int                   advec_diff_dof_idx,
    const int                   curv_dof_idx,
    const int                   curv_quad_idx)
  {
    curv_adaflo_params.dof_index_ls        = advec_diff_dof_idx;
    curv_adaflo_params.dof_index_curvature = curv_dof_idx; //@ todo
    curv_adaflo_params.dof_index_normal    = curv_dof_idx;
    curv_adaflo_params.quad_index          = curv_quad_idx;
    curv_adaflo_params.epsilon =
      parameters.reinit.interface_thickness_parameter.value / parameters.get_n_subdivisions();
    curv_adaflo_params.approximate_projections = false; //@ todo
    curv_adaflo_params.curvature_correction    = parameters.curv.do_curvature_correction;
    verbosity_level                            = parameters.curv.verbosity_level;
    // curv_adaflo_params.filter_parameter = parameters.ls.normal_vec.filter_parameter; //@
    // todo
  }

  template <int dim, typename number>
  void
  CurvatureOperationAdaflo<dim, number>::initialize_vectors()
  {
    /**
     * initialize advected field dof vectors
     */
    scratch_data.initialize_dof_vector(curvature_field, curv_adaflo_params.dof_index_curvature);
    scratch_data.initialize_dof_vector(normal_vec_dummy, curv_adaflo_params.dof_index_curvature);
    /**
     * initialize vectors for the solution of the linear system
     */
    scratch_data.initialize_dof_vector(rhs, curv_adaflo_params.dof_index_curvature);
  }

  template class CurvatureOperationAdaflo<1, double>;
  template class CurvatureOperationAdaflo<2, double>;
  template class CurvatureOperationAdaflo<3, double>;
} // namespace MeltPoolDG::LevelSet
#endif
