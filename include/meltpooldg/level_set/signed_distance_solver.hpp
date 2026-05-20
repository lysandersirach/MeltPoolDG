// SPDX-FileCopyrightText: Copyright (c) 2025-2026 The Lethe Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later

#pragma once
#include <deal.II/base/bounding_box.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/table_handler.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/q_collection.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/matrix_free/fe_point_evaluation.h>

#include <deal.II/non_matching/fe_immersed_values.h>
#include <deal.II/non_matching/fe_values.h>
#include <deal.II/non_matching/mesh_classifier.h>
#include <deal.II/non_matching/quadrature_generator.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace MeltPoolDG::LevelSet
{
  using namespace dealii;

  /**
   * @brief Verbosity levels used to control console output.
   */
  enum class Verbosity
  {
    quiet,
    verbose
  };

  /**
   * @brief Return the sign of a value.
   *
   * @tparam T Numeric type of the input value.
   *
   * @param[in] val Value for which the sign is evaluated.
   *
   * @return -1 if @p val is negative, 0 if it is zero, and 1 if it is positive.
   */
  template <typename T>
  [[nodiscard]] constexpr int
  sgn(const T val) noexcept
  {
    return (static_cast<T>(0) < val) - (val < static_cast<T>(0));
  }

  /**
   * @brief Print a string surrounded by delimiter lines.
   *
   * @param[in] pcout Parallel conditional output stream.
   * @param[in] expression String to print.
   * @param[in] delimiter Character used to build the delimiter lines.
   */
  inline void
  announce_string(const ConditionalOStream &pcout,
                  const std::string        &expression,
                  const char                delimiter = '-')
  {
    pcout << std::string(expression.size() + 1, delimiter) << std::endl;
    pcout << expression << std::endl;
    pcout << std::string(expression.size() + 1, delimiter) << std::endl;
  }

  /**
   * @brief Compute an equivalent cell diameter from a cell measure.
   *
   * The diameter is obtained from the diameter of a disk (2D) or sphere (3D)
   * with the same measure and is scaled by the finite element degree.
   *
   * @tparam dim An integer that denotes the dimension of the space in which
   * the problem is solved.
   *
   * @param[in] cell_measure Measure of the cell, i.e., area in 2D or volume in 3D.
   * @param[in] fe_degree Finite element degree used to scale the diameter.
   *
   * @return Equivalent cell diameter.
   */
  template <int dim>
  inline double
  compute_cell_diameter(const double cell_measure, const unsigned int fe_degree)
  {
    if constexpr (dim == 2)
      return std::sqrt(4. * cell_measure / dealii::numbers::PI) / fe_degree;
    else if constexpr (dim == 3)
      return std::cbrt(6. * cell_measure / dealii::numbers::PI) / fe_degree;
    else
      AssertThrow(false, ExcMessage("Only dim=2 and dim=3 are supported."));
    return 0.0;
  }

  /**
   * @brief Build a map from mesh vertex indices to the active cells that
   * contain them.
   *
   * Only locally owned and ghost cells are inserted in the map.
   *
   * @tparam dim An integer that denotes the dimension of the space in which
   * the problem is solved.
   *
   * @param[in] dof_handler DoFHandler associated with the mesh.
   * @param[in,out] vertices_cell_map Map filled with the cells attached to
   * each vertex index.
   */
  template <int dim>
  void
  vertices_cell_mapping(
    const DoFHandler<dim> &dof_handler,
    std::map<unsigned int, std::set<typename DoFHandler<dim>::active_cell_iterator>>
      &vertices_cell_map)
  {
    vertices_cell_map.clear();
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned() || cell->is_ghost())
        for (unsigned int i = 0; i < GeometryInfo<dim>::vertices_per_cell; ++i)
          vertices_cell_map[cell->vertex_index(i)].insert(cell);
  }

  /**
   * @brief Find all active cells sharing at least one vertex with a given cell.
   *
   * @tparam dim An integer that denotes the dimension of the space in which
   * the problem is solved.
   *
   * @param[in] vertices_cell_map Map from vertex indices to active cells.
   * @param[in] cell Cell for which the surrounding cells are requested.
   *
   * @return Vector containing the cells sharing a vertex with @p cell.
   */
  template <int dim>
  std::vector<typename DoFHandler<dim>::active_cell_iterator>
  find_cells_around_cell(
    std::map<unsigned int, std::set<typename DoFHandler<dim>::active_cell_iterator>>
                                                         &vertices_cell_map,
    const typename DoFHandler<dim>::active_cell_iterator &cell)
  {
    std::set<typename DoFHandler<dim>::active_cell_iterator> neighbors_cells;
    for (unsigned int i = 0; i < GeometryInfo<dim>::vertices_per_cell; ++i)
      {
        const unsigned int v_index = cell->vertex_index(i);
        neighbors_cells.insert(vertices_cell_map[v_index].begin(),
                               vertices_cell_map[v_index].end());
      }

    return {neighbors_cells.begin(), neighbors_cells.end()};
  }

  /**
   * @brief Compute the shortest distance between a point and an interface
   * segment or triangle.
   *
   * In 2D, @p triangle is interpreted as a line segment described by its first
   * two points. In 3D, it is interpreted as a triangle described by its first
   * three points.
   *
   * @tparam dim An integer that denotes the dimension of the space in which
   * the problem is solved.
   *
   * @param[in] triangle Points describing the segment in 2D or triangle in 3D.
   * @param[in] point Point from which the distance is computed.
   *
   * @return Shortest Euclidean distance between @p point and the interface
   * segment or triangle.
   */
  template <int dim>
  double
  find_point_triangle_distance(const std::vector<Point<dim>> &triangle, const Point<dim> &point)
  {
    double distance = 0.0;

    const Point<dim> &point_0 = triangle[0];
    const Point<dim> &point_1 = triangle[1];

    if constexpr (dim == 3)
      {
        const Point<dim> &point_2 = triangle[2];

        const Tensor<1, dim> e_0 = point_1 - point_0;
        const Tensor<1, dim> e_1 = point_2 - point_0;

        const double a   = e_0.norm_square();
        const double b   = scalar_product(e_0, e_1);
        const double c   = e_1.norm_square();
        const double det = a * c - b * b;

        const Tensor<1, dim> vector_to_plane = point_0 - point;

        const double d = scalar_product(e_0, vector_to_plane);
        const double e = scalar_product(e_1, vector_to_plane);

        double s = b * e - c * d;
        double t = b * d - a * e;

        if (s + t <= det)
          {
            if (s < 0)
              {
                if (t < 0)
                  {
                    if (d < 0)
                      {
                        t = 0;
                        if (-d >= a)
                          s = 1;
                        else
                          s = -d / a;
                      }
                    else
                      {
                        s = 0;
                        if (e >= 0)
                          t = 0;
                        else if (-e >= c)
                          t = 1;
                        else
                          t = e / c;
                      }
                  }
                else
                  {
                    s = 0;
                    if (e >= 0)
                      t = 0;
                    else if (-e >= c)
                      t = 1;
                    else
                      t = -e / c;
                  }
              }
            else if (t < 0)
              {
                t = 0;
                if (d >= 0)
                  s = 0;
                else if (-d >= a)
                  s = 1;
                else
                  s = -d / a;
              }
            else
              {
                const double inv_det = 1. / det;
                s *= inv_det;
                t *= inv_det;
              }
          }
        else
          {
            if (s < 0)
              {
                const double tmp0 = b + d;
                const double tmp1 = c + e;
                if (tmp1 > tmp0)
                  {
                    const double numer = tmp1 - tmp0;
                    const double denom = a - 2 * b + c;
                    if (numer >= denom)
                      s = 1;
                    else
                      s = numer / denom;
                    t = 1 - s;
                  }
                else
                  {
                    s = 0;
                    if (tmp1 <= 0)
                      t = 1;
                    else if (e >= 0)
                      t = 0;
                    else
                      t = -e / c;
                  }
              }
            else if (t < 0)
              {
                const double tmp0 = b + e;
                const double tmp1 = a + d;
                if (tmp1 > tmp0)
                  {
                    const double numer = tmp1 - tmp0;
                    const double denom = a - 2 * b + c;
                    if (numer >= denom)
                      t = 1;
                    else
                      t = numer / denom;
                    s = 1 - t;
                  }
                else
                  {
                    t = 0;
                    if (tmp1 <= 0)
                      s = 1;
                    else if (d >= 0)
                      s = 0;
                    else
                      s = -d / a;
                  }
              }
            else
              {
                const double numer = (c + e) - (b + d);
                if (numer <= 0)
                  s = 0;
                else
                  {
                    const double denom = a - 2 * b + c;
                    if (numer >= denom)
                      s = 1;
                    else
                      s = numer / denom;
                  }
                t = 1 - s;
              }
          }

        const Point<dim> pt_in_triangle = point_0 + s * e_0 + t * e_1;
        distance                        = pt_in_triangle.distance(point);
      }

    if constexpr (dim == 2)
      {
        const Tensor<1, dim> d     = point_1 - point_0;
        const double         t_bar = d * (point - point_0) / (d.norm() * d.norm() + 1e-12);

        if (t_bar <= 0.0)
          distance = (point - point_0).norm();
        else if (t_bar >= 1.0)
          distance = (point - point_1).norm();
        else
          distance = (point - (point_0 + t_bar * d)).norm();
      }

    return distance;
  }

  /**
   * @brief Scalar function defined by the DoF values of a single cell.
   *
   * Based on the CellWiseFunction and RefSpaceFEFieldFunction of deal.II.
   *
   * @tparam dim An integer that denotes the dimension of the space in which
   * the problem is solved.
   * @tparam VectorType The vector type of the solution vector.
   * @tparam FEType The finite element type used to discretize the problem.
   */
  template <int dim, typename VectorType = Vector<double>, typename FEType = FE_Q<dim>>
  class CellWiseFunction : public Function<dim>
  {
  public:
    /**
     * @brief Constructor.
     *
     * @param[in] p_fe_degree Finite element degree discretizing the field to
     * convert to a CellWiseFunction.
     */
    CellWiseFunction(const unsigned int p_fe_degree)
      : fe(p_fe_degree)
      , n_cell_wise_dofs(fe.dofs_per_cell)
    {}

    /**
     * @brief Set the active cell values used for subsequent evaluations.
     *
     * @param[in] in_local_dof_values Cell DoF values.
     */
    void
    set_active_cell(const VectorType &in_local_dof_values)
    {
      cell_dof_values = in_local_dof_values;
    }

    /**
     * @brief Return the value of the function at a point in the reference cell.
     *
     * @param[in] point Coordinates of the point in the reference cell.
     * @param[in] component Index of the component for which the value is computed.
     *
     * @return Value of the function, or requested component, at @p point.
     */
    double
    value(const Point<dim> &point, const unsigned int component = 0) const override
    {
      double value = 0.;
      for (unsigned int i = 0; i < n_cell_wise_dofs; ++i)
        value += cell_dof_values[i] * fe.shape_value_component(i, point, component);
      return value;
    }

    /**
     * @brief Return the gradient of the specified component at a point in the
     * reference cell.
     *
     * @param[in] point Coordinates of the point in the reference cell.
     * @param[in] component Index of the component for which the gradient is
     * computed.
     *
     * @return Gradient of the specified component at @p point.
     */
    Tensor<1, dim>
    gradient(const Point<dim> &point, const unsigned int component = 0) const override
    {
      Tensor<1, dim> gradient;
      for (unsigned int i = 0; i < n_cell_wise_dofs; ++i)
        gradient += cell_dof_values[i] * fe.shape_grad_component(i, point, component);
      return gradient;
    }

    /**
     * @brief Return the Hessian of the specified component at a point in the
     * reference cell.
     *
     * @param[in] point Coordinates of the point in the reference cell.
     * @param[in] component Index of the component for which the Hessian is
     * computed.
     *
     * @return Hessian of the specified component at @p point.
     */
    SymmetricTensor<2, dim>
    hessian(const Point<dim> &point, const unsigned int component = 0) const override
    {
      Tensor<2, dim> hessian;
      for (unsigned int i = 0; i < n_cell_wise_dofs; ++i)
        hessian += cell_dof_values[i] * fe.shape_grad_grad_component(i, point, component);
      return symmetrize(hessian);
    }

  private:
    /// Finite element discretizing the field of interest.
    FEType fe;
    /// Number of DoFs per element.
    unsigned int n_cell_wise_dofs;
    /// Value of the field at the DoFs of the active cell.
    VectorType cell_dof_values;
  };

  /**
   * @brief Compute the volume enclosed by the zero level of a level-set field
   * inside a cell.
   *
   * The inside volume is defined by negative values of the level-set field.
   *
   * @tparam dim An integer that denotes the dimension of the space in which
   * the problem is solved.
   *
   * @param[in] fe_point_evaluation FEPointEvaluation used for point-wise
   * evaluations.
   * @param[in] cell Cell for which the volume is computed.
   * @param[in] cell_dof_level_set_values Cell DoF values of the level-set field.
   * @param[in] corr Correction applied uniformly to the DoF values.
   * @param[in] n_quad_points Number of quadrature points for the volume
   * integration faces.
   *
   * @return Cell-wise volume enclosed by the level-set field.
   */
  template <int dim>
  double
  compute_cell_wise_volume(FEPointEvaluation<1, dim> &fe_point_evaluation,
                           const typename DoFHandler<dim>::active_cell_iterator &cell,
                           Vector<double>     cell_dof_level_set_values,
                           const double       corr,
                           const unsigned int n_quad_points);

  /**
   * @brief Integrate the surface area of the zero level of a level-set field
   * and the volume enclosed by it.
   *
   * The inside volume is defined by negative values of the level-set field.
   *
   * @tparam dim An integer that denotes the dimension of the space in which
   * the problem is solved.
   * @tparam VectorType The vector type of the solution vector.
   *
   * @param[in] dof_handler DoFHandler associated with the triangulation on
   * which the volume is computed.
   * @param[in] fe Finite element.
   * @param[in] level_set_vector_relevant_copy Level-set vector.
   *
   * @return Volume enclosed by the zero level and its surface area. The volume
   * is the first value and the surface area is the second.
   */
  template <int dim, typename VectorType>
  std::pair<double, double>
  integrate_volume_and_surface(const DoFHandler<dim>    &dof_handler,
                               const FiniteElement<dim> &fe,
                               const VectorType         &level_set_vector_relevant_copy);

  /**
   * @brief Integrate the surface area of a given level of a level-set field
   * and the volume enclosed by it.
   *
   * The zero contour is defined with respect to @p iso_level by shifting the
   * values of the level-set field.
   *
   * @tparam dim An integer that denotes the dimension of the space in which
   * the problem is solved.
   * @tparam VectorType The vector type of the solution vector.
   *
   * @param[in] dof_handler DoFHandler associated with the triangulation on
   * which the volume is computed.
   * @param[in] fe Finite element.
   * @param[in] level_set_vector Level-set vector.
   * @param[in] iso_level Given level of the level-set field enclosing the
   * volume of interest.
   *
   * @return Volume and surface area enclosed by @p iso_level. The volume is
   * the first value and the surface area is the second.
   */
  template <int dim, typename VectorType>
  std::pair<double, double>
  integrate_volume_and_surface(const DoFHandler<dim>    &dof_handler,
                               const FiniteElement<dim> &fe,
                               const VectorType         &level_set_vector,
                               const double              iso_level);

  /**
   * @brief Reconstruct the interface defined by a given level of a level-set
   * field in the domain.
   *
   * @tparam dim An integer that denotes the dimension of the space in which
   * the problem is solved.
   * @tparam VectorType The vector type of the solution vector.
   *
   * @param[in] mapping Mapping of the domain.
   * @param[in] dof_handler DoFHandler associated with the triangulation for
   * which the interface is reconstructed.
   * @param[in] fe Finite element.
   * @param[in] level_set_vector Level-set vector.
   * @param[in] iso_level Given level of the level-set field defining the
   * interface of interest.
   * @param[in,out] interface_reconstruction_vertices Cell-wise map of the
   * reconstructed interface vertices.
   * @param[in,out] interface_reconstruction_cells Cell-wise map of the
   * reconstructed interface cells.
   * @param[in,out] intersected_dofs Set of DoFs belonging to intersected
   * cells.
   */
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
    std::set<types::global_dof_index> &intersected_dofs);

  /**
   * @brief Interface to build patches of the interface reconstruction vertices
   * for visualization.
   *
   * It reproduces the same structure as particle visualization.
   *
   * @tparam dim An integer that denotes the number of spatial dimensions.
   */
  template <int dim>
  class InterfaceReconstructionDataOut : public dealii::DataOutInterface<0, dim>
  {
  public:
    /**
     * @brief Build patches from interface reconstruction vertices for
     * visualization.
     *
     * @param[in] interface_reconstruction_vertices Cell-wise map of the
     * reconstructed interface vertices.
     */
    void
    build_patches(const std::map<types::global_cell_index, std::vector<Point<dim>>>
                    &interface_reconstruction_vertices)
    {
      for (const auto &cell : interface_reconstruction_vertices)
        for (const Point<dim> &vertex : cell.second)
          {
            DataOutBase::Patch<0, dim> patch;
            patch.vertices[0] = vertex;
            patches.push_back(patch);
          }
    }

  private:
    /**
     * @brief Implementation of the corresponding function of the base class.
     */
    const std::vector<DataOutBase::Patch<0, dim>> &
    get_patches() const override
    {
      return patches;
    }

    /**
     * @brief Implementation of the corresponding function of the base class.
     */
    std::vector<std::string>
    get_dataset_names() const override
    {
      return dataset_names;
    }

    /// Output information filled by build_patches() and written by the base class.
    std::vector<DataOutBase::Patch<0, dim>> patches;
    /// List of field names for all data components stored in patches.
    std::vector<std::string> dataset_names;
  };

  /**
   * @brief Solver to compute the signed distance from a given level of a
   * level-set field.
   *
   * It is based on the geometric mass-preserving redistancing scheme proposed
   * by Ausas, Dari and Buscaglia (2011).
   *
   * @tparam dim An integer that denotes the dimension of the space in which
   * the problem is solved.
   * @tparam VectorType The vector type of the level-set vector.
   */
  template <int dim, typename VectorType = LinearAlgebra::distributed::Vector<double>>
  class SignedDistanceSolver
  {
  public:
    /**
     * @brief Constructor.
     *
     * @param[in] background_triangulation Shared pointer to the triangulation
     * of the domain.
     * @param[in] background_fe Shared pointer to the finite element
     * discretizing the domain.
     * @param[in] p_max_distance Maximum reinitialization distance value.
     * @param[in] p_iso_level Iso-level before scaling from which the signed
     * distance is computed.
     * @param[in] p_scaling Scaling factor applied to the input level-set field.
     * @param[in] p_verbosity Verbosity level.
     */
    SignedDistanceSolver(
      std::shared_ptr<parallel::DistributedTriangulationBase<dim>> background_triangulation,
      const FiniteElement<dim>                                    &background_fe,
      const double                                                 p_max_distance,
      const double                                                 p_iso_level,
      const double                                                 p_scaling   = 1.0,
      const Verbosity                                              p_verbosity = Verbosity::quiet);

    /**
     * @brief Constructor.
     *
     * @param[in] background_dof_handler Shared pointer to the DoFHandler
     * of the domain.
     * @param[in] p_max_distance Maximum reinitialization distance value.
     * @param[in] p_iso_level Iso-level before scaling from which the signed
     * distance is computed.
     * @param[in] p_scaling Scaling factor applied to the input level-set field.
     * @param[in] p_verbosity Verbosity level.
     */
    SignedDistanceSolver(std::shared_ptr<DoFHandler<dim>> background_dof_handler,
                         const double                     p_max_distance,
                         const double                     p_iso_level,
                         const double                     p_scaling   = 1.0,
                         const Verbosity                  p_verbosity = Verbosity::quiet);

    /**
     * @brief Initialize the degrees of freedom and associated memory.
     */
    void
    setup_dofs();

    /**
     * @brief Set the level-set field from the background mesh solver.
     *
     * @param[in] background_dof_handler DoFHandler corresponding to the
     * level-set field solver.
     * @param[in] background_level_set_vector Level-set solution vector from
     * the background solver.
     */
    void
    set_level_set_from_background_mesh(const DoFHandler<dim> &background_dof_handler,
                                       const VectorType      &background_level_set_vector);

    /**
     * @brief Solve for the signed distance from the given level of the
     * level-set vector.
     */
    void
    solve();

    /**
     * @brief Get the computed signed distance field.
     *
     * @return Reference to the vector storing the computed signed distance
     * values.
     */
    VectorType &
    get_signed_distance();

    /**
     * @brief Get the computed signed distance field.
     *
     * @return Reference to the vector storing the computed signed distance
     * values.
     */
    const VectorType &
    get_signed_distance() const;

    /**
     * @brief Get the DoFHandler used by the signed distance solver.
     *
     * @return Constant reference to the solver DoFHandler.
     */
    const DoFHandler<dim> &
    get_dof_handler() const
    {
      return *dof_handler;
    }

    /**
     * @brief Output the interface reconstruction used for signed distance
     * computations.
     *
     * @param[in] filename Name of the output file.
     */
    void
    output_interface_reconstruction(const std::string &filename) const;

  private:
    /**
     * @brief Zero the ghost DoF entries of solution vectors to gain write
     * access to ghost elements.
     */
    void
    zero_out_ghost_values() const;

    /**
     * @brief Update ghost DoF entries of solution vectors to gain read access
     * to ghost elements.
     */
    void
    update_ghost_values() const;

    /**
     * @brief Exchange the distance field and keep the minimum value across
     * processors with compress(VectorOperation::min).
     */
    void
    exchange_distance();

    /**
     * @brief Initialize the distance vectors to the maximum distance.
     */
    void
    initialize_distance();

    /**
     * @brief Compute brute-force geometric distances between the reconstructed
     * interface and DoFs of intersected cells.
     */
    void
    compute_first_neighbors_distance();

    /**
     * @brief Compute marching-method geometric distances between the
     * reconstructed interface and remaining DoFs.
     */
    void
    compute_second_neighbors_distance();

    /**
     * @brief Compute signed_distance from the unsigned distance and the sign
     * of the signed-distance entries.
     */
    void
    compute_signed_distance_from_distance();

    /**
     * @brief Compute the cell-wise volume correction matching the volume
     * enclosed by the given level of the level-set field in each cell.
     */
    void
    compute_cell_wise_volume_correction();

    /**
     * @brief Correct the global volume to match the volume enclosed by the
     * given level of the level-set field.
     */
    void
    conserve_global_volume();

    /**
     * @brief Set the map of local ids of opposite DoFs for each face.
     */
    inline void
    set_face_opposite_dofs_map();

    /**
     * @brief Set the map of local ids of face DoFs for each face.
     */
    inline void
    set_face_dofs_map();

    /**
     * @brief Return the local ids of DoFs opposite to a given face.
     *
     * @param[in] local_face_id Local id of the face in the cell.
     * @param[out] local_opposite_dofs Vector containing the local ids of the
     * opposite DoFs.
     */
    inline void
    get_face_opposite_dofs(unsigned int               local_face_id,
                           std::vector<unsigned int> &local_opposite_dofs) const;

    /**
     * @brief Return the local ids of the DoFs located on a given face.
     *
     * @param[in] local_face_id Local id of the face in the cell.
     * @param[out] local_dofs Vector containing the local ids of the face DoFs.
     */
    inline void
    get_face_local_dofs(unsigned int local_face_id, std::vector<unsigned int> &local_dofs) const;

    /**
     * @brief Return the face transformation Jacobian.
     *
     * This is required because the distance minimization problem is solved in
     * the reference face space.
     *
     * @param[in] cell_transformation_jac Cell transformation Jacobian.
     * @param[in] local_face_id Local id of the face.
     * @param[out] face_transformation_jac Face transformation Jacobian.
     */
    inline void
    get_face_transformation_jacobian(const DerivativeForm<1, dim, dim> &cell_transformation_jac,
                                     const unsigned int                 local_face_id,
                                     LAPACKFullMatrix<double> &face_transformation_jac) const;

    /**
     * @brief Transform a point from a reference face to the reference cell.
     *
     * @param[in] x_ref_face Point in the reference face.
     * @param[in] local_face_id Local id of the face.
     *
     * @return Point in the reference cell.
     */
    inline Point<dim>
    transform_ref_face_point_to_ref_cell(const Point<dim - 1> &x_ref_face,
                                         const unsigned int    local_face_id) const;

    /**
     * @brief Compute the residual of the distance minimization problem in the
     * reference face space.
     *
     * @param[in] x_n_to_x_I_real Vector from x_n to x_I in real space.
     * @param[in] distance_gradient Gradient of the distance at x_n.
     * @param[in] transformation_jac Transformation Jacobian of the face.
     * @param[out] residual_ref Residual in reference face space.
     */
    inline void
    compute_residual(const Tensor<1, dim>           &x_n_to_x_I_real,
                     const Tensor<1, dim>           &distance_gradient,
                     const LAPACKFullMatrix<double> &transformation_jac,
                     Tensor<1, dim - 1>             &residual_ref) const;

    /**
     * @brief Transform a Newton correction from the reference face to the
     * reference cell.
     *
     * @param[in] correction_ref_face Newton correction in the reference face.
     * @param[in] local_face_id Local id of the face.
     *
     * @return Correction vector in the reference cell.
     */
    inline Tensor<1, dim>
    transform_ref_face_correction_to_ref_cell(const Vector<double> &correction_ref_face,
                                              const unsigned int    local_face_id) const;

    /**
     * @brief Compute the analytical Jacobian of the distance minimization
     * problem in the reference face space.
     *
     * @param[in] x_n_to_x_I_real_p1 Vector from x_n to x_I at the current
     * Newton iterate in real space.
     * @param[in] transformation_jacobian Face transformation Jacobian.
     * @param[in] face_local_dof_values Values of the DoFs on the face.
     * @param[out] jacobian_matrix Jacobian matrix of the minimization problem.
     */
    inline void
    compute_analytical_jacobian(const Tensor<1, dim>           &x_n_to_x_I_real_p1,
                                const LAPACKFullMatrix<double> &transformation_jacobian,
                                const std::vector<double>      &face_local_dof_values,
                                LAPACKFullMatrix<double>       &jacobian_matrix);

    /**
     * @brief Compute the propagated distance d(x_I) = d(x_n) + ||x_I - x_n||.
     *
     * @param[in] x_n_to_x_I_real Vector from x_n to x_I in real space.
     * @param[in] distance Distance value at x_n.
     *
     * @return Distance between x_I and the interface.
     */
    inline double
    compute_distance(const Tensor<1, dim> &x_n_to_x_I_real, const double distance) const
    {
      return distance + x_n_to_x_I_real.norm();
    }
    /// Flag to indicate whether the DoFHandler is provided externally or built
    /// by the solver.
    bool is_external_dof_handler = false;
    /// DoFHandler describing the signed distance problem.
    std::shared_ptr<DoFHandler<dim>> dof_handler;
    /// Finite element discretizing the signed distance problem.
    const FiniteElement<dim> &fe;
    /// Mapping between the real and reference spaces.
    std::shared_ptr<MappingQ<dim>> mapping;

    /// Maximum redistancing distance.
    const double max_distance;
    /// Iso-level describing the interface from which the signed distance is computed.
    const double iso_level;
    /// Scaling factor applied to the input level-set field.
    const double scaling;
    /// Verbosity level.
    const Verbosity verbosity;

    /// Parallel output stream.
    ConditionalOStream pcout;

    /// Set of locally owned DoFs.
    IndexSet locally_owned_dofs;
    /// Set of locally relevant DoFs.
    IndexSet locally_relevant_dofs;
    /// Set of locally active DoFs.
    IndexSet locally_active_dofs;

    /// Level-set field coming from the background solver.
    VectorType level_set;

    /// Solution vector of the signed distance without ghost values.
    LinearAlgebra::distributed::Vector<double> signed_distance;
    /// Solution vector of the signed distance with ghost values.
    LinearAlgebra::distributed::Vector<double> signed_distance_with_ghost;
    /// Solution vector of the unsigned distance without ghost values.
    LinearAlgebra::distributed::Vector<double> distance;
    /// Solution vector of the unsigned distance with ghost values.
    LinearAlgebra::distributed::Vector<double> distance_with_ghost;
    /// Correction applied to match the cell-wise volume enclosed by the level set.
    LinearAlgebra::distributed::Vector<double> volume_correction;

    /// Hanging node constraints.
    AffineConstraints<double> constraints;

    std::map<types::global_cell_index, std::vector<Point<dim>>> interface_reconstruction_vertices;
    std::map<types::global_cell_index, std::vector<CellData<dim == 1 ? 1 : dim - 1>>>
      interface_reconstruction_cells;
    /// Set of DoFs belonging to intersected cells.
    std::set<types::global_dof_index> intersected_dofs;

    /// Map from face ids to local ids of opposite DoFs.
    std::map<unsigned int, std::vector<unsigned int>> face_opposite_dofs_map;
    /// Map from face ids to local ids of face DoFs.
    std::map<unsigned int, std::vector<unsigned int>> face_dofs_map;

    /// Temporary Hessian matrix used in analytical Jacobian computations.
    LAPACKFullMatrix<double> hessian_matrix;
    /// Temporary product of the Hessian and the transformation Jacobian.
    LAPACKFullMatrix<double> H_x_transformation_jacobian;
  };

} // namespace MeltPoolDG::LevelSet
