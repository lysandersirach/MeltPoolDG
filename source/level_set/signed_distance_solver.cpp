#include <deal.II/base/exception_macros.h>

#include <meltpooldg/level_set/signed_distance_solver.hpp>

namespace MeltPoolDG::LevelSet
{
  using namespace dealii;

  template <int dim>
  double
  compute_cell_wise_volume(FEPointEvaluation<1, dim> &fe_point_evaluation,
                           const typename DoFHandler<dim>::active_cell_iterator &cell,
                           Vector<double>     cell_dof_level_set_values,
                           const double       corr,
                           const unsigned int n_quad_points)
  {
    const unsigned int n_dofs = cell_dof_level_set_values.size();

    const BoundingBox<dim> unit_box = create_unit_bounding_box<dim>();
    CellWiseFunction<dim>  signed_distance_function(cell->get_fe().degree);

    hp::QCollection<1> q_collection;
    q_collection.push_back(QGauss<1>(n_quad_points));

    NonMatching::QuadratureGenerator<dim> quadrature_generator(q_collection);

    for (unsigned int j = 0; j < n_dofs; ++j)
      cell_dof_level_set_values[j] += corr;

    signed_distance_function.set_active_cell(cell_dof_level_set_values);
    quadrature_generator.generate(signed_distance_function, unit_box);

    const Quadrature<dim> inside_quadrature = quadrature_generator.get_inside_quadrature();

    if (inside_quadrature.size() == 0)
      return 0.0;

    fe_point_evaluation.reinit(cell, inside_quadrature.get_points());

    double inside_cell_volume = 0.0;
    for (unsigned int q = 0; q < inside_quadrature.size(); ++q)
      inside_cell_volume +=
        fe_point_evaluation.jacobian(q).determinant() * inside_quadrature.weight(q);

    return inside_cell_volume;
  }

  template <int dim, typename VectorType>
  std::pair<double, double>
  integrate_volume_and_surface(const DoFHandler<dim>    &dof_handler,
                               const FiniteElement<dim> &fe,
                               const VectorType         &level_set_vector_relevant_copy)
  {
    const MPI_Comm mpi_communicator = dof_handler.get_mpi_communicator();

    NonMatching::MeshClassifier<dim> mesh_classifier(dof_handler, level_set_vector_relevant_copy);
    mesh_classifier.reclassify();

    const hp::FECollection<dim> fe_collection(fe);
    const QGauss<1>             quadrature_1D(fe.degree + 1);

    NonMatching::RegionUpdateFlags region_update_flags;
    region_update_flags.inside  = update_JxW_values;
    region_update_flags.surface = update_JxW_values;

    NonMatching::FEValues<dim> non_matching_fe_values(fe_collection,
                                                      quadrature_1D,
                                                      region_update_flags,
                                                      mesh_classifier,
                                                      dof_handler,
                                                      level_set_vector_relevant_copy);

    double volume  = 0.0;
    double surface = 0.0;

    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          non_matching_fe_values.reinit(cell);

          const std::optional<FEValues<dim>> &inside_fe_values =
            non_matching_fe_values.get_inside_fe_values();

          if (inside_fe_values)
            for (const unsigned int q : inside_fe_values->quadrature_point_indices())
              volume += inside_fe_values->JxW(q);

          const std::optional<NonMatching::FEImmersedSurfaceValues<dim>> &surface_fe_values =
            non_matching_fe_values.get_surface_fe_values();

          if (surface_fe_values)
            for (const unsigned int q : surface_fe_values->quadrature_point_indices())
              surface += surface_fe_values->JxW(q);
        }

    volume  = Utilities::MPI::sum(volume, mpi_communicator);
    surface = Utilities::MPI::sum(surface, mpi_communicator);

    return {volume, surface};
  }

  template <int dim, typename VectorType>
  std::pair<double, double>
  integrate_volume_and_surface(const DoFHandler<dim>    &dof_handler,
                               const FiniteElement<dim> &fe,
                               const VectorType         &level_set_vector,
                               const double              iso_level)
  {
    const MPI_Comm mpi_communicator = dof_handler.get_mpi_communicator();

    VectorType level_set_vector_owned_copy(dof_handler.locally_owned_dofs(), mpi_communicator);
    level_set_vector_owned_copy = level_set_vector;
    level_set_vector_owned_copy.add(-iso_level);

    VectorType level_set_vector_relevant_copy(dof_handler.locally_owned_dofs(),
                                              DoFTools::extract_locally_relevant_dofs(dof_handler),
                                              mpi_communicator);
    level_set_vector_relevant_copy = level_set_vector_owned_copy;

    return integrate_volume_and_surface(dof_handler, fe, level_set_vector_relevant_copy);
  }

  template <int dim, typename VectorType>
  void
  reconstruct_interface(
    const Mapping<dim>                                          &mapping,
    const DoFHandler<dim>                                       &dof_handler,
    const FiniteElement<dim>                                    &fe,
    const VectorType                                            &level_set_vector,
    const double                                                 iso_level,
    std::map<types::global_cell_index, std::vector<Point<dim>>> &interface_reconstruction_vertices,
    std::map<types::global_cell_index, std::vector<CellData<dim == 1 ? 1 : dim - 1>>>
                                      &interface_reconstruction_cells,
    std::set<types::global_dof_index> &intersected_dofs)
  {
    GridTools::MarchingCubeAlgorithm<dim, VectorType> marching_cube(mapping, fe, 1, 1e-10);
    const unsigned int                                dofs_per_cell = fe.n_dofs_per_cell();

    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned() || cell->is_ghost())
        {
          std::vector<Point<dim>>                       surface_vertices;
          std::vector<CellData<dim == 1 ? 1 : dim - 1>> surface_cells;

          marching_cube.process_cell(
            cell, level_set_vector, iso_level, surface_vertices, surface_cells);

          if (!surface_vertices.empty())
            {
              const unsigned int cell_index                 = cell->global_active_cell_index();
              interface_reconstruction_vertices[cell_index] = surface_vertices;
              interface_reconstruction_cells[cell_index]    = surface_cells;

              if (cell->is_locally_owned())
                {
                  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);
                  cell->get_dof_indices(dof_indices);
                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    intersected_dofs.insert(dof_indices[i]);
                }
            }
        }
  }

  template <int dim, typename VectorType>
  SignedDistanceSolver<dim, VectorType>::SignedDistanceSolver(
    std::shared_ptr<parallel::DistributedTriangulationBase<dim>> background_triangulation,
    const FiniteElement<dim>                                    &background_fe,
    const double                                                 p_max_distance,
    const double                                                 p_iso_level,
    const double                                                 p_scaling,
    const Verbosity                                              p_verbosity)
    : dof_handler(std::make_shared<DoFHandler<dim>>(*background_triangulation))
    , fe(background_fe)
    , max_distance(p_max_distance)
    , iso_level(p_iso_level * p_scaling)
    , scaling(p_scaling)
    , verbosity(p_verbosity)
    , pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    AssertThrow(background_fe.degree == 1,
                ExcMessage("Signed distance solver is only implemented for degree=1"));

    mapping = std::make_shared<MappingQ<dim>>(fe.degree);
    set_face_opposite_dofs_map();
    set_face_dofs_map();


    hessian_matrix              = LAPACKFullMatrix<double>(dim, dim);
    H_x_transformation_jacobian = LAPACKFullMatrix<double>(dim, dim - 1);
  }

  template <int dim, typename VectorType>
  SignedDistanceSolver<dim, VectorType>::SignedDistanceSolver(
    std::shared_ptr<DoFHandler<dim>> background_dof_handler,
    const double                     p_max_distance,
    const double                     p_iso_level,
    const double                     p_scaling,
    const Verbosity                  p_verbosity)
    : is_external_dof_handler(true)
    , dof_handler(background_dof_handler)
    , fe(background_dof_handler->get_fe())
    , max_distance(p_max_distance)
    , iso_level(p_iso_level * p_scaling)
    , scaling(p_scaling)
    , verbosity(p_verbosity)
    , pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    AssertThrow(fe.degree == 1,
                ExcMessage("Signed distance solver is only implemented for degree=1"));

    mapping = std::make_shared<MappingQ<dim>>(fe.degree);
    set_face_opposite_dofs_map();
    set_face_dofs_map();

    hessian_matrix              = LAPACKFullMatrix<double>(dim, dim);
    H_x_transformation_jacobian = LAPACKFullMatrix<double>(dim, dim - 1);
  }

  template <int dim, typename VectorType>
  void
  SignedDistanceSolver<dim, VectorType>::setup_dofs()
  {
    const MPI_Comm mpi_communicator = dof_handler->get_mpi_communicator();

    if (not is_external_dof_handler)
      dof_handler->distribute_dofs(fe);

    locally_owned_dofs    = dof_handler->locally_owned_dofs();
    locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(*dof_handler);
    locally_active_dofs   = DoFTools::extract_locally_active_dofs(*dof_handler);

    level_set.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);

    signed_distance.reinit(locally_owned_dofs, locally_active_dofs, mpi_communicator);
    signed_distance_with_ghost.reinit(locally_owned_dofs, locally_active_dofs, mpi_communicator);
    distance.reinit(locally_owned_dofs, locally_active_dofs, mpi_communicator);
    distance_with_ghost.reinit(locally_owned_dofs, locally_active_dofs, mpi_communicator);
    volume_correction.reinit(locally_owned_dofs, locally_active_dofs, mpi_communicator);

    constraints.clear();
    constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(*dof_handler, constraints);
    constraints.close();
  }

  template <int dim, typename VectorType>
  void
  SignedDistanceSolver<dim, VectorType>::set_level_set_from_background_mesh(
    const DoFHandler<dim> &background_dof_handler,
    const VectorType      &background_level_set_vector)
  {
    const MPI_Comm mpi_communicator = dof_handler->get_mpi_communicator();
    VectorType     tmp_local_level_set(locally_owned_dofs, mpi_communicator);

    FETools::interpolate(background_dof_handler,
                         background_level_set_vector,
                         *dof_handler,
                         constraints,
                         tmp_local_level_set);

    tmp_local_level_set *= scaling;
    level_set = tmp_local_level_set;
  }

  template <int dim, typename VectorType>
  void
  SignedDistanceSolver<dim, VectorType>::solve()
  {
    if (verbosity != Verbosity::quiet)
      announce_string(pcout, "Signed Distance Solver");

    zero_out_ghost_values();

    interface_reconstruction_vertices.clear();
    interface_reconstruction_cells.clear();
    intersected_dofs.clear();

    initialize_distance();

    reconstruct_interface(*mapping,
                          *dof_handler,
                          fe,
                          level_set,
                          iso_level,
                          interface_reconstruction_vertices,
                          interface_reconstruction_cells,
                          intersected_dofs);

    compute_first_neighbors_distance();
    compute_signed_distance_from_distance();

    compute_cell_wise_volume_correction();
    conserve_global_volume();

    compute_second_neighbors_distance();

    compute_signed_distance_from_distance();

    update_ghost_values();
  }

  template <int dim, typename VectorType>
  void
  SignedDistanceSolver<dim, VectorType>::initialize_distance()
  {
    for (auto p : locally_active_dofs)
      {
        distance(p)            = max_distance;
        distance_with_ghost(p) = max_distance;

        const double sign_ls          = sgn(level_set(p) - iso_level);
        signed_distance(p)            = max_distance * sign_ls;
        signed_distance_with_ghost(p) = max_distance * sign_ls;
      }
  }

  template <int dim, typename VectorType>
  VectorType &
  SignedDistanceSolver<dim, VectorType>::get_signed_distance()
  {
    return signed_distance_with_ghost;
    // ALTERNATIVE FROM LETHE:
    //
    // const MPI_Comm mpi_communicator = dof_handler.get_mpi_communicator();
    // VectorType     tmp_local_level_set(locally_owned_dofs, mpi_communicator);

    // for (auto p : locally_owned_dofs)
    // tmp_local_level_set(p) = signed_distance(p);

    // level_set = tmp_local_level_set;
    // return level_set;
  }

  template <int dim, typename VectorType>
  const VectorType &
  SignedDistanceSolver<dim, VectorType>::get_signed_distance() const
  {
    return signed_distance_with_ghost;
  }

  template <int dim, typename VectorType>
  void
  SignedDistanceSolver<dim, VectorType>::zero_out_ghost_values() const
  {
    signed_distance.zero_out_ghost_values();
    signed_distance_with_ghost.zero_out_ghost_values();
    distance.zero_out_ghost_values();
    distance_with_ghost.zero_out_ghost_values();
    volume_correction.zero_out_ghost_values();
  }

  template <int dim, typename VectorType>
  void
  SignedDistanceSolver<dim, VectorType>::update_ghost_values() const
  {
    signed_distance.update_ghost_values();
    signed_distance_with_ghost.update_ghost_values();
    distance.update_ghost_values();
    distance_with_ghost.update_ghost_values();
    volume_correction.update_ghost_values();
  }

  template <int dim, typename VectorType>
  void
  SignedDistanceSolver<dim, VectorType>::exchange_distance()
  {
    distance.compress(VectorOperation::min);
    distance.update_ghost_values();
    distance_with_ghost = distance;
    distance.zero_out_ghost_values();

    for (auto p : locally_active_dofs)
      distance(p) = distance_with_ghost(p);
  }

  template <int dim, typename VectorType>
  inline void
  SignedDistanceSolver<dim, VectorType>::set_face_opposite_dofs_map()
  {
    if constexpr (dim == 2)
      {
        face_opposite_dofs_map[0] = {1, 3};
        face_opposite_dofs_map[1] = {0, 2};
        face_opposite_dofs_map[2] = {2, 3};
        face_opposite_dofs_map[3] = {0, 1};
      }

    if constexpr (dim == 3)
      {
        face_opposite_dofs_map[0] = {1, 3, 5, 7};
        face_opposite_dofs_map[1] = {0, 2, 4, 6};
        face_opposite_dofs_map[2] = {2, 3, 6, 7};
        face_opposite_dofs_map[3] = {0, 1, 4, 5};
        face_opposite_dofs_map[4] = {4, 5, 6, 7};
        face_opposite_dofs_map[5] = {0, 1, 2, 3};
      }
  }

  template <int dim, typename VectorType>
  inline void
  SignedDistanceSolver<dim, VectorType>::set_face_dofs_map()
  {
    if constexpr (dim == 2)
      {
        face_dofs_map[0] = {0, 2};
        face_dofs_map[1] = {1, 3};
        face_dofs_map[2] = {0, 1};
        face_dofs_map[3] = {2, 3};
      }

    if constexpr (dim == 3)
      {
        face_dofs_map[0] = {0, 2, 4, 6};
        face_dofs_map[1] = {1, 3, 5, 7};
        face_dofs_map[2] = {0, 1, 4, 5};
        face_dofs_map[3] = {2, 3, 6, 7};
        face_dofs_map[4] = {0, 1, 2, 3};
        face_dofs_map[5] = {4, 5, 6, 7};
      }
  }

  template <int dim, typename VectorType>
  inline void
  SignedDistanceSolver<dim, VectorType>::get_face_opposite_dofs(
    unsigned int               local_face_id,
    std::vector<unsigned int> &local_opposite_dofs) const
  {
    local_opposite_dofs = face_opposite_dofs_map.at(local_face_id);
  }

  template <int dim, typename VectorType>
  inline void
  SignedDistanceSolver<dim, VectorType>::get_face_local_dofs(
    unsigned int               local_face_id,
    std::vector<unsigned int> &local_dofs) const
  {
    local_dofs = face_dofs_map.at(local_face_id);
  }

  template <int dim, typename VectorType>
  inline void
  SignedDistanceSolver<dim, VectorType>::get_face_transformation_jacobian(
    const DerivativeForm<1, dim, dim> &cell_transformation_jac,
    const unsigned int                 local_face_id,
    LAPACKFullMatrix<double>          &face_transformation_jac) const
  {
    for (unsigned int i = 0; i < dim; ++i)
      {
        int k = 0;
        for (unsigned int j = 0; j < dim; ++j)
          {
            if (local_face_id / 2 == j)
              continue;
            if (k < static_cast<int>(dim - 1))
              face_transformation_jac(i, k) = cell_transformation_jac[i][j];
            k += 1;
          }
      }
  }

  template <int dim, typename VectorType>
  inline Point<dim>
  SignedDistanceSolver<dim, VectorType>::transform_ref_face_point_to_ref_cell(
    const Point<dim - 1> &x_ref_face,
    const unsigned int    local_face_id) const
  {
    Point<dim>   x_ref_cell;
    unsigned int j = 0;
    for (unsigned int i = 0; i < dim; ++i)
      {
        if (local_face_id / 2 == i)
          {
            x_ref_cell[i] = double(local_face_id % 2);
            continue;
          }
        x_ref_cell[i] = x_ref_face[j++];
      }
    return x_ref_cell;
  }

  template <int dim, typename VectorType>
  inline void
  SignedDistanceSolver<dim, VectorType>::compute_residual(
    const Tensor<1, dim>           &x_n_to_x_I_real,
    const Tensor<1, dim>           &distance_gradient,
    const LAPACKFullMatrix<double> &transformation_jac,
    Tensor<1, dim - 1>             &residual_ref) const
  {
    residual_ref.clear();
    const Tensor<1, dim> residual_real =
      distance_gradient - (1.0 / x_n_to_x_I_real.norm()) * x_n_to_x_I_real;

    for (unsigned int i = 0; i < dim - 1; ++i)
      for (unsigned int j = 0; j < dim; ++j)
        residual_ref[i] += transformation_jac(j, i) * residual_real[j];
  }

  template <int dim, typename VectorType>
  inline Tensor<1, dim>
  SignedDistanceSolver<dim, VectorType>::transform_ref_face_correction_to_ref_cell(
    const Vector<double> &correction_ref_face,
    const unsigned int    local_face_id) const
  {
    Tensor<1, dim> correction_ref_cell;
    unsigned int   j = 0;
    for (unsigned int i = 0; i < dim; ++i)
      {
        if (local_face_id / 2 == i)
          correction_ref_cell[i] = 0.0;
        else
          correction_ref_cell[i] = correction_ref_face[j++];
      }
    return correction_ref_cell;
  }

  template <int dim, typename VectorType>
  inline void
  SignedDistanceSolver<dim, VectorType>::compute_analytical_jacobian(
    const Tensor<1, dim>           &x_n_to_x_I_real_p1,
    const LAPACKFullMatrix<double> &transformation_jacobian,
    const std::vector<double>      &face_local_dof_values,
    LAPACKFullMatrix<double>       &jacobian_matrix)
  {
    const double norm      = x_n_to_x_I_real_p1.norm();
    const double norm_inv  = 1.0 / norm;
    const double norm3_inv = norm_inv * norm_inv * norm_inv;

    for (int i = 0; i < dim; ++i)
      {
        hessian_matrix(i, i) =
          -(x_n_to_x_I_real_p1[i] * x_n_to_x_I_real_p1[i]) * norm3_inv + norm_inv;

        for (int j = i + 1; j < dim; ++j)
          {
            const double h_ij    = -(x_n_to_x_I_real_p1[i] * x_n_to_x_I_real_p1[j]) * norm3_inv;
            hessian_matrix(i, j) = h_ij;
            hessian_matrix(j, i) = h_ij;
          }
      }

    for (int i = 0; i < dim; ++i)
      for (int j = 0; j < dim - 1; ++j)
        {
          double matrix_ij = 0;
          for (int k = 0; k < dim; ++k)
            matrix_ij += hessian_matrix(i, k) * transformation_jacobian(k, j);
          H_x_transformation_jacobian(i, j) = matrix_ij;
        }

    for (int i = 0; i < dim - 1; ++i)
      for (int j = i; j < dim - 1; ++j)
        {
          double matrix_ij = 0;
          for (int k = 0; k < dim; ++k)
            matrix_ij += transformation_jacobian(k, i) * H_x_transformation_jacobian(k, j);
          jacobian_matrix(i, j) = matrix_ij;
          jacobian_matrix(j, i) = matrix_ij;
        }

    if constexpr (dim == 3)
      {
        const double off_diag_H = face_local_dof_values[0] - face_local_dof_values[1] -
                                  face_local_dof_values[2] + face_local_dof_values[3];
        jacobian_matrix(0, 1) += off_diag_H;
        jacobian_matrix(1, 0) += off_diag_H;
      }
  }

  template <int dim, typename VectorType>
  void
  SignedDistanceSolver<dim, VectorType>::compute_first_neighbors_distance()
  {
    std::map<types::global_dof_index, Point<dim>> dof_support_points =
      DoFTools::map_dofs_to_support_points(*mapping, *dof_handler);

    for (auto &intersected_cell : interface_reconstruction_cells)
      {
        const unsigned int cell_index = intersected_cell.first;

        std::vector<Point<dim>> surface_vertices = interface_reconstruction_vertices.at(cell_index);
        std::vector<CellData<dim == 1 ? 1 : dim - 1>> surface_cells = intersected_cell.second;

        Triangulation<dim == 1 ? 1 : dim - 1, dim> surface_triangulation;
        surface_triangulation.create_triangulation(surface_vertices, surface_cells, {});

        for (const auto &intersected_dof : intersected_dofs)
          {
            const Point<dim> y = dof_support_points.at(intersected_dof);

            for (const auto &surface_cell : surface_triangulation.active_cell_iterators())
              {
                const unsigned int      surface_cell_n_vertices = surface_cell->n_vertices();
                std::vector<Point<dim>> surface_cell_vertices(surface_cell_n_vertices);
                for (unsigned int p = 0; p < surface_cell_n_vertices; ++p)
                  surface_cell_vertices[p] = surface_cell->vertex(p);

                const double D = find_point_triangle_distance(surface_cell_vertices, y);

                distance(intersected_dof) =
                  std::min(std::abs(distance(intersected_dof)), std::abs(D));
              }
          }
      }

    exchange_distance();
  }

  template <int dim, typename VectorType>
  void
  SignedDistanceSolver<dim, VectorType>::compute_second_neighbors_distance()
  {
    const MPI_Comm mpi_communicator = dof_handler->get_mpi_communicator();

    const unsigned int dofs_per_cell             = fe.n_dofs_per_cell();
    unsigned int       faces_per_cell            = (dim == 3 ? 6 : 4);
    unsigned int       n_opposite_dofs_per_faces = (dim == 3 ? 4 : 2);
    unsigned int       n_dofs_per_faces          = (dim == 3 ? 4 : 2);

    std::vector<types::global_dof_index> dof_indices(dofs_per_cell);
    std::vector<double>                  cell_dof_values(dofs_per_cell);
    std::vector<unsigned int>            face_opposite_dofs(n_opposite_dofs_per_faces);
    std::vector<unsigned int>            face_local_dofs(n_dofs_per_faces);
    std::vector<double>                  face_local_dof_values(n_dofs_per_faces);

    Point<dim>     x_I_real;
    Tensor<1, dim> x_n_to_x_I_real;
    Tensor<1, dim> correction;
    Point<dim>     x_n_p1_ref;
    Point<dim - 1> ref_face_center_point;
    ref_face_center_point(0) = 0.5;
    if constexpr (dim == 3)
      ref_face_center_point(1) = 0.5;

    std::vector<Point<dim>> x_n_ref_vector(n_opposite_dofs_per_faces);
    std::vector<Point<dim>> x_n_real_vector(n_opposite_dofs_per_faces);
    Tensor<1, dim>          distance_gradients;

    DerivativeForm<1, dim, dim> cell_transformation_jacobian;
    LAPACKFullMatrix<double>    face_transformation_jacobian(dim, dim - 1);

    LAPACKFullMatrix<double> jacobian_matrix(dim - 1, dim - 1);
    jacobian_matrix.set_property(LAPACKSupport::general);
    Vector<double>      residual_n_vec(dim - 1);
    std::vector<double> correction_norm(n_opposite_dofs_per_faces, 1.0);
    std::vector<int>    outside_check(n_opposite_dofs_per_faces);

    std::map<types::global_dof_index, Point<dim>> dof_support_points =
      DoFTools::map_dofs_to_support_points(*mapping, *dof_handler);

    FEPointEvaluation<1, dim> fe_point_evaluation(*mapping,
                                                  fe,
                                                  update_gradients | update_jacobians);
    FEPointEvaluation<1, dim> fe_point_evaluation_values_only(*mapping, fe, update_values);

    bool change = true;

    std::map<unsigned int, std::set<typename DoFHandler<dim>::active_cell_iterator>>
      vertices_to_cell;
    vertices_cell_mapping(*dof_handler, vertices_to_cell);

    std::set<typename DoFHandler<dim>::active_cell_iterator> corona_cell;
    for (const auto &cell : dof_handler->active_cell_iterators())
      if (cell->is_locally_owned())
        {
          const unsigned int cell_index = cell->global_active_cell_index();
          if (interface_reconstruction_vertices.contains(cell_index))
            {
              auto active_neighbors = find_cells_around_cell<dim>(vertices_to_cell, cell);
              corona_cell.insert(active_neighbors.begin(), active_neighbors.end());
            }
        }

    int count = 0;
    while (change)
      {
        if (verbosity != Verbosity::quiet)
          pcout << "Solving signed distance of layer " << count << std::endl;

        change = false;

        for (const auto &cell : dof_handler->active_cell_iterators())
          if (cell->is_locally_owned())
            {
              cell->get_dof_values(distance_with_ghost,
                                   cell_dof_values.begin(),
                                   cell_dof_values.end());
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                if (std::abs(cell_dof_values[i]) < max_distance)
                  {
                    corona_cell.insert(cell);
                    break;
                  }
            }

        for (const auto &cell : corona_cell)
          if (cell->is_locally_owned())
            {
              const unsigned int cell_index = cell->global_active_cell_index();
              if (interface_reconstruction_vertices.contains(cell_index))
                continue;

              cell->get_dof_indices(dof_indices);
              cell->get_dof_values(distance_with_ghost,
                                   cell_dof_values.begin(),
                                   cell_dof_values.end());

              for (unsigned int j = 0; j < faces_per_cell; ++j)
                {
                  get_face_opposite_dofs(j, face_opposite_dofs);
                  get_face_local_dofs(j, face_local_dofs);

                  for (unsigned int i = 0; i < n_opposite_dofs_per_faces; ++i)
                    {
                      x_n_ref_vector[i] =
                        transform_ref_face_point_to_ref_cell(ref_face_center_point, j);
                      correction_norm[i] = 1.0;
                      outside_check[i]   = 0;
                    }

                  for (unsigned int i = 0; i < n_dofs_per_faces; ++i)
                    face_local_dof_values[i] = cell_dof_values[face_local_dofs[i]];

                  constexpr double tol           = 1e-9;
                  unsigned int     newton_it     = 0;
                  constexpr int    newton_max_it = 100;

                  while (*std::max_element(correction_norm.begin(), correction_norm.end()) > tol &&
                         newton_it < static_cast<unsigned int>(newton_max_it))
                    {
                      fe_point_evaluation.reinit(cell, x_n_ref_vector);
                      fe_point_evaluation.evaluate(cell_dof_values, EvaluationFlags::gradients);

                      for (unsigned int i = 0; i < n_opposite_dofs_per_faces; ++i)
                        {
                          if (correction_norm[i] < tol)
                            continue;

                          const unsigned int local_face_opposite_dof = face_opposite_dofs[i];

                          if (intersected_dofs.contains(dof_indices[local_face_opposite_dof]))
                            {
                              correction_norm[i] = 0;
                              continue;
                            }

                          x_I_real = dof_support_points.at(dof_indices[local_face_opposite_dof]);

                          x_n_real_vector[i]           = fe_point_evaluation.quadrature_point(i);
                          distance_gradients           = fe_point_evaluation.get_gradient(i);
                          cell_transformation_jacobian = fe_point_evaluation.jacobian(i);
                          get_face_transformation_jacobian(cell_transformation_jacobian,
                                                           j,
                                                           face_transformation_jacobian);

                          x_n_to_x_I_real = x_I_real - x_n_real_vector[i];

                          compute_analytical_jacobian(x_n_to_x_I_real,
                                                      face_transformation_jacobian,
                                                      face_local_dof_values,
                                                      jacobian_matrix);

                          Tensor<1, dim - 1> residual_n;
                          compute_residual(x_n_to_x_I_real,
                                           distance_gradients,
                                           face_transformation_jacobian,
                                           residual_n);

                          residual_n.unroll(residual_n_vec.begin(), residual_n_vec.end());
                          residual_n_vec *= -1.0;

                          jacobian_matrix.compute_lu_factorization();
                          jacobian_matrix.solve(residual_n_vec);
                          jacobian_matrix.reinit(dim - 1, dim - 1);

                          correction_norm[i] = residual_n_vec.l2_norm();

                          correction = transform_ref_face_correction_to_ref_cell(residual_n_vec, j);

                          x_n_p1_ref        = x_n_ref_vector[i] + correction;
                          double relaxation = 1.0;
                          bool   check      = false;

                          for (int k = 0; k < dim; ++k)
                            if (x_n_p1_ref[k] > 1.0 + tol || x_n_p1_ref[k] < 0.0 - tol)
                              {
                                check = true;
                                if (correction[k] > tol)
                                  relaxation =
                                    std::min((1.0 - x_n_ref_vector[i][k]) / (correction[k] + tol),
                                             relaxation);
                                else if (correction[k] < -tol)
                                  relaxation =
                                    std::min((0.0 - x_n_ref_vector[i][k]) / (correction[k] + tol),
                                             relaxation);
                              }

                          if (check)
                            outside_check[i] += 1;

                          if (outside_check[i] > 3)
                            {
                              correction_norm[i] = 0.0;
                              correction         = 0.0;
                            }

                          x_n_p1_ref        = x_n_ref_vector[i] + relaxation * correction;
                          x_n_ref_vector[i] = x_n_p1_ref;
                        }

                      newton_it += 1;
                    }

                  fe_point_evaluation_values_only.reinit(cell, x_n_ref_vector);
                  fe_point_evaluation_values_only.evaluate(cell_dof_values,
                                                           EvaluationFlags::values);

                  for (unsigned int i = 0; i < n_opposite_dofs_per_faces; ++i)
                    {
                      const unsigned int local_face_opposite_dof = face_opposite_dofs[i];

                      x_I_real = dof_support_points.at(dof_indices[local_face_opposite_dof]);

                      x_n_real_vector[i] = fe_point_evaluation_values_only.quadrature_point(i);
                      x_n_to_x_I_real    = x_I_real - x_n_real_vector[i];

                      const double distance_value_at_x_n =
                        fe_point_evaluation_values_only.get_value(i);

                      const double approx_distance =
                        compute_distance(x_n_to_x_I_real, distance_value_at_x_n);

                      constexpr double distance_tol = 1e-8;

                      const auto gdof = dof_indices[local_face_opposite_dof];

                      if (distance(gdof) > approx_distance + distance_tol)
                        // if (distance.locally_owned_elements().is_element(gdof) and
                        // distance(gdof) > approx_distance + distance_tol)
                        {
                          change                                         = true;
                          distance(dof_indices[local_face_opposite_dof]) = approx_distance;
                        }
                    }
                }
            }

        exchange_distance();
        change = Utilities::MPI::logical_or(change, mpi_communicator);
        count += 1;
      }
    constraints.distribute(distance);
  }

  template <int dim, typename VectorType>
  void
  SignedDistanceSolver<dim, VectorType>::compute_signed_distance_from_distance()
  {
    for (auto p : locally_active_dofs)
      signed_distance(p) = distance(p) * sgn(signed_distance_with_ghost(p));

    signed_distance.update_ghost_values();
    signed_distance_with_ghost = signed_distance;
    signed_distance.zero_out_ghost_values();

    for (auto p : locally_active_dofs)
      signed_distance(p) = signed_distance_with_ghost(p);
  }

  template <int dim, typename VectorType>
  void
  SignedDistanceSolver<dim, VectorType>::compute_cell_wise_volume_correction()
  {
    FEPointEvaluation<1, dim> fe_point_evaluation(*mapping,
                                                  fe,
                                                  update_jacobians | update_JxW_values);

    const unsigned int dofs_per_cell        = fe.n_dofs_per_cell();
    double             n_cells_per_dofs_inv = (dim == 3 ? 1.0 / 8.0 : 1.0 / 4.0);

    volume_correction = 0.0;

    for (const auto &cell : dof_handler->active_cell_iterators())
      if (cell->is_locally_owned())
        {
          const unsigned int cell_index = cell->global_active_cell_index();

          if (!interface_reconstruction_vertices.contains(cell_index))
            continue;

          Vector<double> cell_level_set_dof_values(dofs_per_cell);
          cell->get_dof_values(level_set,
                               cell_level_set_dof_values.begin(),
                               cell_level_set_dof_values.end());

          const double targeted_cell_volume = compute_cell_wise_volume(
            fe_point_evaluation, cell, cell_level_set_dof_values, -iso_level, fe.degree + 1);

          Vector<double> cell_dof_values(dofs_per_cell);
          cell->get_dof_values(signed_distance_with_ghost,
                               cell_dof_values.begin(),
                               cell_dof_values.end());

          const double cell_volume = cell->measure();
          const double cell_size   = compute_cell_diameter<dim>(cell_volume, fe.degree);

          double inside_cell_volume_nm1 = 0.0;
          double inside_cell_volume_n   = 0.0;
          double delta_volume_nm1       = 0.0;
          double delta_volume_n         = 0.0;
          double delta_volume_prime     = 0.0;

          double eta_nm1 = 0.0;
          double eta_n   = 1e-6 * cell_size;
          double eta_np1 = 0.0;

          inside_cell_volume_nm1 = compute_cell_wise_volume(
            fe_point_evaluation, cell, cell_dof_values, eta_nm1, fe.degree + 1);
          delta_volume_nm1 = targeted_cell_volume - inside_cell_volume_nm1;

          const double initial_inside_cell_volume = inside_cell_volume_nm1;

          constexpr double tol = 1e-12;
          if (inside_cell_volume_nm1 < tol * cell_volume ||
              inside_cell_volume_nm1 > (cell_volume - tol * cell_volume))
            continue;

          unsigned int       secant_it     = 0;
          const unsigned int secant_max_it = 20;
          double             secant_update = 1.0;

          while (std::abs(secant_update) > tol &&
                 std::abs(delta_volume_nm1) > tol * initial_inside_cell_volume &&
                 secant_it < secant_max_it)
            {
              if (inside_cell_volume_nm1 < tol * cell_size ||
                  inside_cell_volume_nm1 > (cell_size - tol * cell_size))
                {
                  eta_n = 0.0;
                  break;
                }

              ++secant_it;

              inside_cell_volume_n = compute_cell_wise_volume(
                fe_point_evaluation, cell, cell_dof_values, eta_n, fe.degree + 1);

              delta_volume_n     = targeted_cell_volume - inside_cell_volume_n;
              delta_volume_prime = (delta_volume_n - delta_volume_nm1) / (eta_n - eta_nm1 + tol);
              secant_update      = -delta_volume_n / (delta_volume_prime + tol);
              eta_np1            = eta_n + secant_update;

              eta_nm1                = eta_n;
              eta_n                  = eta_np1;
              inside_cell_volume_nm1 = inside_cell_volume_n;
              delta_volume_nm1       = delta_volume_n;
            }

          if (secant_it >= secant_max_it)
            eta_n = 0.0;

          std::vector<types::global_dof_index> dof_indices(dofs_per_cell);
          cell->get_dof_indices(dof_indices);

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            volume_correction(dof_indices[i]) += eta_n * n_cells_per_dofs_inv;
        }

    volume_correction.compress(VectorOperation::add);
    volume_correction.update_ghost_values();
  }

  template <int dim, typename VectorType>
  void
  SignedDistanceSolver<dim, VectorType>::conserve_global_volume()
  {
    const MPI_Comm mpi_communicator = dof_handler->get_mpi_communicator();

    double global_volume, surface;
    std::tie(global_volume, surface) =
      integrate_volume_and_surface(*dof_handler, fe, level_set, iso_level);

    double global_volume_nm1         = 0.0;
    double global_volume_n           = 0.0;
    double global_delta_volume_nm1   = 0.0;
    double global_delta_volume_n     = 0.0;
    double global_delta_volume_prime = 0.0;

    double C_nm1 = 1.0;
    double C_n   = 1e-6 * global_volume;
    double C_np1 = 0.0;

    LinearAlgebra::distributed::Vector<double> signed_distance_0(signed_distance_with_ghost);
    signed_distance_0.add(C_nm1, volume_correction);
    signed_distance_0.update_ghost_values();

    std::tie(global_volume_nm1, surface) =
      integrate_volume_and_surface(*dof_handler, fe, signed_distance_0, 0.0);

    global_delta_volume_nm1      = global_volume - global_volume_nm1;
    const double global_volume_0 = global_volume_nm1;

    constexpr double       tol           = 1e-12;
    unsigned int           secant_it     = 0;
    constexpr unsigned int secant_max_it = 20;
    double                 secant_update = 1.0;

    while (std::abs(secant_update) > tol &&
           std::abs(global_delta_volume_nm1) > tol * global_volume_0 && secant_it < secant_max_it)
      {
        ++secant_it;

        LinearAlgebra::distributed::Vector<double> signed_distance_n(signed_distance_with_ghost);
        signed_distance_n.add(C_n, volume_correction);
        signed_distance_n.update_ghost_values();

        std::tie(global_volume_n, surface) =
          integrate_volume_and_surface(*dof_handler, fe, signed_distance_n, 0.0);

        global_delta_volume_n = global_volume - global_volume_n;
        global_delta_volume_prime =
          (global_delta_volume_n - global_delta_volume_nm1) / (C_n - C_nm1 + tol);
        secant_update = -global_delta_volume_n / (global_delta_volume_prime + tol);

        C_np1 = C_n + secant_update;
        C_nm1 = C_n;
        C_n   = C_np1;

        global_volume_nm1       = global_volume_n;
        global_delta_volume_nm1 = global_delta_volume_n;
      }

    if (secant_it >= secant_max_it)
      C_n = 0.0;

    if (verbosity != Verbosity::quiet && Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
      {
        TableHandler volume_conservation_table;
        volume_conservation_table.declare_column("Initial volume");
        volume_conservation_table.add_value("Initial volume", global_volume);
        volume_conservation_table.set_scientific("Initial volume", true);

        volume_conservation_table.declare_column("Volume after redistanciation");
        volume_conservation_table.add_value("Volume after redistanciation", global_volume_n);
        volume_conservation_table.set_scientific("Volume after redistanciation", true);

        volume_conservation_table.declare_column("Remaining error on the volume");
        volume_conservation_table.add_value("Remaining error on the volume", global_delta_volume_n);
        volume_conservation_table.set_scientific("Remaining error on the volume", true);

        volume_conservation_table.write_text(std::cout);
      }

    signed_distance_with_ghost.add(C_n, volume_correction);
    signed_distance_with_ghost.update_ghost_values();

    for (auto p : locally_active_dofs)
      {
        signed_distance(p) = signed_distance_with_ghost(p);
        distance(p)        = std::abs(signed_distance(p));
      }

    exchange_distance();
  }

  template <int dim, typename VectorType>
  void
  SignedDistanceSolver<dim, VectorType>::output_interface_reconstruction(
    const std::string &filename) const
  {
    InterfaceReconstructionDataOut<dim> reconstruction_data_out;
    reconstruction_data_out.build_patches(interface_reconstruction_vertices);
    std::ofstream out(filename);
    reconstruction_data_out.write_vtu(out);
  }

  template class SignedDistanceSolver<1, LinearAlgebra::distributed::Vector<double>>;
  template class SignedDistanceSolver<2, LinearAlgebra::distributed::Vector<double>>;
  template class SignedDistanceSolver<3, LinearAlgebra::distributed::Vector<double>>;

  template std::pair<double, double>
  integrate_volume_and_surface(const DoFHandler<2> &,
                               const FiniteElement<2> &,
                               const LinearAlgebra::distributed::Vector<double> &);
  template std::pair<double, double>
  integrate_volume_and_surface(const DoFHandler<3> &,
                               const FiniteElement<3> &,
                               const LinearAlgebra::distributed::Vector<double> &);

  template std::pair<double, double>
  integrate_volume_and_surface(const DoFHandler<2> &,
                               const FiniteElement<2> &,
                               const LinearAlgebra::distributed::Vector<double> &,
                               const double);
  template std::pair<double, double>
  integrate_volume_and_surface(const DoFHandler<3> &,
                               const FiniteElement<3> &,
                               const LinearAlgebra::distributed::Vector<double> &,
                               const double);

  template void
  reconstruct_interface(const Mapping<2> &,
                        const DoFHandler<2> &,
                        const FiniteElement<2> &,
                        const LinearAlgebra::distributed::Vector<double> &,
                        const double,
                        std::map<types::global_cell_index, std::vector<Point<2>>> &,
                        std::map<types::global_cell_index, std::vector<CellData<1>>> &,
                        std::set<types::global_dof_index> &);

  template void
  reconstruct_interface(const Mapping<3> &,
                        const DoFHandler<3> &,
                        const FiniteElement<3> &,
                        const LinearAlgebra::distributed::Vector<double> &,
                        const double,
                        std::map<types::global_cell_index, std::vector<Point<3>>> &,
                        std::map<types::global_cell_index, std::vector<CellData<2>>> &,
                        std::set<types::global_dof_index> &);

  template double
  compute_cell_wise_volume<2>(FEPointEvaluation<1, 2> &,
                              const DoFHandler<2>::active_cell_iterator &,
                              Vector<double>,
                              const double,
                              const unsigned int);

  template double
  compute_cell_wise_volume<3>(FEPointEvaluation<1, 3> &,
                              const DoFHandler<3>::active_cell_iterator &,
                              Vector<double>,
                              const double,
                              const unsigned int);
} // namespace MeltPoolDG::LevelSet
