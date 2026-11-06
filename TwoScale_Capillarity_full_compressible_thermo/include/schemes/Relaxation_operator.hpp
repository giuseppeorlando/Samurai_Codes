// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Author: Giuseppe Orlando, 2026
//
#pragma once

#include "flux_base.hpp"

#define DEBUG_RELAXATION

namespace EquationData {
  // Use auxiliary indices also to distinguish bewteen the type of relaxation
  static constexpr std::size_t PRESSURE_EQUILIBRIUM = 0;
  static constexpr std::size_t LOCAL_LAPLACE        = 1;
}

namespace samurai {
  using namespace EquationData;

  /**
   * Implementation of a relaxation operator
   */
  template<class Field>
  class RelaxationOperator {
  public:
    using Number = typename Field::value_type; // Define the shortcut for the arithmetic type

    using cfg = samurai::LocalCellSchemeConfig<SchemeType::NonLinear, Field, Field>;

    /**
     * Class constructor
     * @param EOS_phase_liq_ liquid equation of state
     * @param EOS_phase_gas_ gas equation of state
     * @param sigma_ surface tension coefficient
     * @param Hmax_ threshold length scale
     * @param kappa_ parameter related to the radius of small-scale droplets
     * @param alpha_d_max_ maximum threshold of small-scale volume fraction
     * @param alpha_l_min_ minimum large-scale volume fraction to identify the mixture region
     * @param alpha_l_max_ maximum large-scale volume fraction to identify the mixture region
     * @param lambda_ bound-preserving parameter
     * @param atol_Newton_ absolute tolerance for dual-time stepping
     * @param rtol_Newton_ relative tolerance for dual-time stepping
     * @param max_Newton_iters_ maximum number of iterations for dual-time stepping
     * @param p_ref_ reference pressure for termination criterion
     * @param mass_transfer_NR_ flag to check whether mass transfer inside relaxation is desired
     * @param freeze_pI_ flag to freeze the interfacial pressure at its value at the first Newton iteration of each cell
     */
    RelaxationOperator(const SG_EOS<Number>& EOS_phase_liq_,
                       const SG_EOS<Number>& EOS_phase_gas_,
                       const Number sigma_,
                       const Number Hmax_,
                       const Number kappa_,
                       const Number alpha_d_max_,
                       const Number alpha_l_min_,
                       const Number alpha_l_max_,
                       const Number lambda_ = static_cast<Number>(0.9),
                       const Number atol_Newton_ = static_cast<Number>(1e-14),
                       const Number rtol_Newton_ = static_cast<Number>(1e-12),
                       const std::size_t max_Newton_iters_ = 60,
                       const Number p_ref_ = static_cast<Number>(1e5),
                       const bool mass_transfer_NR_ = true,
                       const bool freeze_pI_ = false);

    /**
     * Perform a Newton step relaxation for a state vector
     * @param H curvature
     * @param dalpha_l variation of volume fraction
     * @param alpha_l volume fraction
     * @param to_be_relaxed auxiliary flag to mark if a field has still to be relaxed or not
     * @param Newton_iterations number of Newton (dual-time stepping) iterations
     * @param grad_alpha_l gradient of large-scale volume fraction (unused here, it may serve in case mass-transfer location changes)
     * @param type_relaxation check if relaxation toward local Laplace law or toward pressure equilibrium
     * @param pI_0 interfacial pressure at the first Newton iteration of each cell (used only if freeze_pI = true)
     */
    template<class Field_Scalar, class Field_Scalar_Unsigned, class Field_Vect>
    auto make_Newton_step_relaxation(const Field_Scalar& H,
                                     Field_Scalar& dalpha_l,
                                     Field_Scalar& alpha_l,
                                     Field_Scalar_Unsigned& to_be_relaxed,
                                     Field_Scalar_Unsigned& Newton_iterations,
                                     Field_Vect& grad_alpha_l,
                                     Field_Scalar_Unsigned& type_relaxation,
                                     Field_Scalar& pI_0);

    /**
     * Perform an instantaneous pressure relaxation p_liq = p_g (linearization method Pelanti based)
     * for a state vector in internal-energy formulation. Valid only for sigma = 0
     * @param alpha_l volume fraction
     */
    template<class Field_Scalar>
    auto make_instantaneous_pressure_relaxation(Field_Scalar& alpha_l);

    /**
     * Set the value of the flag to check whether relaxation has been applied
     * @param global_relaxation_applied flag to check whether relaxation has been applied
     */
    inline void set_relaxation_applied(const bool global_relaxation_applied);

    /**
     * Get the value of the flag to check whether relaxation has been applied
     * @return global_relaxation_applied flag to check whether relaxation has been applied
     */
    inline bool get_relaxation_applied() const;

    /**
     * Set the value of the flag to check whether mass transfer inside relaxation has to be done or not
     * @param mass_transfer_NR_ flag to check whether mass transfer is desired inside relaxation
     */
    inline void set_mass_transfer_NR(const bool mass_transfer_NR_);

  protected:
    const SG_EOS<Number>& EOS_phase_liq;
    const SG_EOS<Number>& EOS_phase_gas;

    const Number sigma; /*!< Surface tension coefficient */

    const Number Hmax;        /*!< Threshold length scale */
    const Number kappa;       /*!< Parameter related to the radius of small-scale droplets */
    const Number alpha_d_max; /*!< Maximum threshold of small-scale volume fraction */
    const Number alpha_l_min; /*!< Minimum large-scale volume fraction to identify the mixture region */
    const Number alpha_l_max; /*!< Maximum large-scale volume fraction to identify the mixture region */

    const Number      lambda;           /*!< Parameter for bound preserving strategy */
    const Number      atol_Newton;      /*!< Absolute tolerance Newton method relaxation */
    const Number      rtol_Newton;      /*!< Relative tolerance Newton method relaxation */
    const std::size_t max_Newton_iters; /*!< Maximum number of Newton iterations */
    const Number      p_ref;            /*!< Reference pressure for termination criterion */

    const bool freeze_pI; /*!< Freeze the interfacial pressure at its value at the first Newton iteration of each cell */

  private:
    bool mass_transfer_NR; /*!< Auxiliary flag to check whether mass transfer inside relaxation is desired */

    bool relaxation_applied; /*!< Auxiliary flag to check whether relaxation has been applied */
  };

  // Constructor with all relevant parameters
  //
  template<class Field>
  RelaxationOperator<Field>::RelaxationOperator(const SG_EOS<Number>& EOS_phase_liq_,
                                                const SG_EOS<Number>& EOS_phase_gas_,
                                                const Number sigma_,
                                                const Number Hmax_,
                                                const Number kappa_,
                                                const Number alpha_d_max_,
                                                const Number alpha_l_min_,
                                                const Number alpha_l_max_,
                                                const Number lambda_,
                                                const Number atol_Newton_,
                                                const Number rtol_Newton_,
                                                const std::size_t max_Newton_iters_,
                                                const Number p_ref_,
                                                const bool mass_transfer_NR_,
                                                const bool freeze_pI_):
    EOS_phase_liq(EOS_phase_liq_), EOS_phase_gas(EOS_phase_gas_), sigma(sigma_),
    Hmax(Hmax_), kappa(kappa_),
    alpha_d_max(alpha_d_max_), alpha_l_min(alpha_l_min_), alpha_l_max(alpha_l_max_),
    lambda(lambda_), atol_Newton(atol_Newton_), rtol_Newton(rtol_Newton_),
    max_Newton_iters(max_Newton_iters_), p_ref(p_ref_),
    freeze_pI(freeze_pI_),
    mass_transfer_NR(mass_transfer_NR_) {}

  // Set the value of the flag to check whether relaxation has been applied
  //
  template<class Field>
  void RelaxationOperator<Field>::set_relaxation_applied(const bool global_relaxation_applied) {
    relaxation_applied = global_relaxation_applied;
  }

  // Get the value of the flag to check whether relaxation has been applied
  //
  template<class Field>
  bool RelaxationOperator<Field>::get_relaxation_applied() const {
    return relaxation_applied;
  }

  // Set the value of the flag to check whether mass transfer inside relaxation has to be done or not
  //
  template<class Field>
  void RelaxationOperator<Field>::set_mass_transfer_NR(const bool mass_transfer_NR_) {
    mass_transfer_NR = mass_transfer_NR_;
  }

  // Implement the contribution of the discrete relaxation operator
  //
  template<class Field>
  template<class Field_Scalar, class Field_Scalar_Unsigned, class Field_Vect>
  auto RelaxationOperator<Field>::make_Newton_step_relaxation(const Field_Scalar& H,
                                                              Field_Scalar& dalpha_l,
                                                              Field_Scalar& alpha_l,
                                                              Field_Scalar_Unsigned& to_be_relaxed,
                                                              Field_Scalar_Unsigned& Newton_iterations,
                                                              Field_Vect& grad_alpha_l,
                                                              Field_Scalar_Unsigned& type_relaxation,
                                                              Field_Scalar& pI_0) {
    auto relaxation_step = samurai::make_cell_based_scheme<typename RelaxationOperator::cfg>();
    relaxation_step.set_name("Relaxation");
    relaxation_step.set_scheme_function([&](samurai::SchemeValue<cfg>& result, const auto& cell, const auto& field)
                                           {
                                             const auto local_field = field[cell];
                                             result = field[cell];

                                             to_be_relaxed[cell] = 0;

                                             const auto H_loc = H[cell];
                                             // Pre-fetch variables used multiple times to exploit possible vectorization
                                             auto alpha_l_loc  = alpha_l[cell];
                                             auto dalpha_l_loc = dalpha_l[cell];

                                             const auto m_l_loc      = local_field(Ml_INDEX);
                                             const auto m_g_loc      = local_field(Mg_INDEX);
                                             const auto m_d_loc      = local_field(Md_INDEX);
                                             const auto rho_z_loc    = local_field(RHO_Z_INDEX);
                                             const auto mliqeliq_loc = local_field(Mliq_Eliq_INDEX);
                                             /* NOTE: This represents the liquid local augmented internal energy */
                                             const auto mgeg_loc     = local_field(Mg_Eg_INDEX);
                                             /* NOTE: This represents the gas local augmented internal energy */

                                             const auto inv_m_l_loc     = static_cast<Number>(1.0)/m_l_loc;
                                             const auto inv_alpha_l_loc = static_cast<Number>(1.0)/alpha_l_loc;

                                             // Update auxiliary values affected by the nonlinear function for which we seek a zero
                                             const auto alpha_d_loc     = alpha_l_loc*m_d_loc*inv_m_l_loc;
                                                                          // TODO: Add a check in case of zero volume fraction
                                             const auto alpha_liq_loc   = alpha_l_loc + alpha_d_loc;
                                             const auto alpha_g_loc     = static_cast<Number>(1.0) - alpha_liq_loc;
                                             const auto inv_alpha_g_loc = static_cast<Number>(1.0)/alpha_g_loc;

                                             const auto m_liq_loc       = m_l_loc + m_d_loc;
                                             const auto rho_liq_loc     = m_liq_loc/alpha_liq_loc;
                                                                          // TODO: Add a check in case of zero volume fraction
                                             const auto inv_rho_liq_loc = static_cast<Number>(1.0)/rho_liq_loc;

                                             const auto Sigma_d_loc = rho_z_loc*std::cbrt(inv_rho_liq_loc*inv_rho_liq_loc);

                                             const auto rho_loc     = m_liq_loc + m_g_loc;
                                             const auto inv_rho_loc = static_cast<Number>(1.0)/rho_loc;

                                             const auto Y_liq_loc   = m_liq_loc*inv_rho_loc;
                                             const auto chi_liq_loc = Y_liq_loc;
                                             const auto e_liq_loc   = mliqeliq_loc/m_liq_loc
                                                                    - sigma*inv_rho_loc*(chi_liq_loc/Y_liq_loc)*Sigma_d_loc;
                                                                    // TODO: Add a check in case of zero volume fraction
                                             const auto p_liq_loc   = EOS_phase_liq.pres_value_Rhoe(rho_liq_loc, e_liq_loc);

                                             const auto rho_g_loc = m_g_loc*inv_alpha_g_loc;
                                                                    // TODO: Add a check in case of zero volume fraction
                                             const auto Y_g_loc   = static_cast<Number>(1.0) - Y_liq_loc;
                                             const auto chi_g_loc = Y_g_loc;
                                             const auto e_g_loc   = mgeg_loc/m_g_loc
                                                                  - sigma*inv_rho_loc*(chi_g_loc/Y_g_loc)*Sigma_d_loc;
                                                                  // TODO: Add a check in case of zero volume fraction
                                             const auto p_g_loc   = EOS_phase_gas.pres_value_Rhoe(rho_g_loc, e_g_loc);

                                             // Prepare for mass transfer if desired
                                             Number H_lim,
                                                    dH;
                                             const auto fac_Ru      = sigma*Hmax*(static_cast<Number>(3.0)/kappa -
                                                                                  static_cast<Number>(1.0));
                                             const auto mom_squared = local_field(RHO_U_INDEX)*local_field(RHO_U_INDEX)
                                                                    + local_field(RHO_U_INDEX + 1)*local_field(RHO_U_INDEX + 1);
                                             const auto mom_dot_vel = mom_squared/rho_loc;
                                             if(!std::isnan(H_loc)) {
                                               type_relaxation[cell] = LOCAL_LAPLACE;

                                               // Compute region where performing inter-scale transfer
                                               H_lim = std::min(H_loc, Hmax);
                                               if(mass_transfer_NR) {
                                                 if(alpha_l_loc > alpha_l_min && alpha_l_loc < alpha_l_max &&
                                                    alpha_d_loc < alpha_d_max &&
                                                    alpha_l_loc*fac_Ru <= static_cast<Number>(0.5)*mom_dot_vel) {
                                                   ;
                                                 }
                                                 else {
                                                   H_lim = H_loc;
                                                 }
                                               }
                                               else {
                                                 H_lim = H_loc;
                                               }

                                               dH = H_loc - H_lim;
                                             }
                                             else {
                                               type_relaxation[cell] = PRESSURE_EQUILIBRIUM;

                                               H_lim = H_loc; // Useless, set to NaN
                                               dH    = static_cast<Number>(0.0);
                                             }

                                             // Compute the nonlinear function for which we seek the zero
                                             // (basically the Laplace law or the pressure equilibrium)
                                             const auto delta_p = p_liq_loc - p_g_loc;
                                             const auto F_LS    = alpha_l_loc*(delta_p - sigma*H_lim);
                                             const auto aux_SS  = static_cast<Number>(2.0/3.0)*sigma*
                                                                  rho_z_loc*std::cbrt(inv_m_l_loc*inv_m_l_loc*inv_alpha_l_loc);
                                                                  // TODO: Add a check in case of zero volume fraction
                                             const auto F_SS    = alpha_d_loc*delta_p - alpha_l_loc*aux_SS;
                                             const auto F       = (type_relaxation[cell] == PRESSURE_EQUILIBRIUM)?
                                                                  delta_p : F_LS + F_SS;

                                             // Perform the relaxation only where really needed
                                             /* NOTE: The stagnation check on dalpha_l is relative to the smallest volume fraction:
                                                      since dalpha_l ~ min(alpha_liq/(rho c^2)_liq, alpha_g/(rho c^2)_g)*delta_p,
                                                      an absolute threshold would stop residual-phase cells
                                                      (e.g. alpha_l ~ 1e-8) after one iteration, possibly far from equilibrium.
                                                      Recall that alpha_g*alpha_l/alpha_liq is the variation of alpha_l
                                                      which empties the gas phase
                                                      A way to avoid it would be to use a relative threshold, such as
                                                      alpha_l_scale = std::min(alpha_l_loc, alpha_g_loc*alpha_l_loc/alpha_liq_loc);
                                                      std::abs(dalpha_l_loc) > rtol_Newton*alpha_l_scale
                                                      but it is probably too restrictive. */
                                             if((std::abs(F) > atol_Newton + rtol_Newton*((type_relaxation[cell] == PRESSURE_EQUILIBRIUM)?
                                                                                          p_ref :
                                                                                          std::min(p_ref, sigma*std::abs(H_lim))) &&
                                                 std::abs(dalpha_l_loc) > atol_Newton) || dH > rtol_Newton*Hmax) {
                                               to_be_relaxed[cell] = 1;
                                               Newton_iterations[cell]++;
                                               relaxation_applied = true;

                                               // Compute the derivative w.r.t large scale volume fraction
                                               const auto c_liq_loc     = EOS_phase_liq.c_value_RhoP(rho_liq_loc, p_liq_loc);
                                               const auto Gamma_liq_loc = EOS_phase_liq.Gruneisen_Rhoe(rho_liq_loc, e_liq_loc);

                                               const auto c_g_loc     = EOS_phase_gas.c_value_RhoP(rho_g_loc, p_g_loc);
                                               const auto Gamma_g_loc = EOS_phase_gas.Gruneisen_Rhoe(rho_g_loc, e_g_loc);

                                               const auto rho_Gamma_jump = (chi_liq_loc/Y_liq_loc)*rho_liq_loc*Gamma_liq_loc
                                                                         - (chi_g_loc/Y_g_loc)*rho_g_loc*Gamma_g_loc;

                                               const auto ddelta_p_dalpha_l = -m_l_loc*inv_alpha_l_loc*inv_alpha_l_loc*
                                                                              (c_liq_loc*c_liq_loc - p_liq_loc/rho_liq_loc*Gamma_liq_loc)
                                                                              -m_g_loc*inv_alpha_g_loc*inv_alpha_g_loc*
                                                                              (c_g_loc*c_g_loc - p_g_loc/rho_g_loc*Gamma_g_loc)*
                                                                              m_liq_loc*inv_m_l_loc
                                                                              -aux_SS*inv_rho_loc*rho_Gamma_jump;
                                                                              // TODO: Add a check in case of zero volume fraction
                                               const auto dF_LS_dalpha_l    = (delta_p - sigma*H_lim)
                                                                            + alpha_l_loc*ddelta_p_dalpha_l;
                                               const auto dF_SS_dalpha_l    = (m_d_loc*inv_m_l_loc)*delta_p
                                                                            + alpha_d_loc*ddelta_p_dalpha_l
                                                                            - static_cast<Number>(2.0/3.0)*aux_SS;
                                                                            // TODO: Add a check in case of zero volume fraction
                                               const auto dF_dalpha_l       = (type_relaxation[cell] == PRESSURE_EQUILIBRIUM)?
                                                                              ddelta_p_dalpha_l : dF_LS_dalpha_l + dF_SS_dalpha_l;

                                               // Compute the pseudo time step starting as initial guess
                                               // from the ideal unmodified Newton method
                                               auto dtau_ov_epsilon = std::numeric_limits<Number>::infinity();

                                               // Bound-preserving condition for m_l, velocity and small-scale volume fraction
                                               const auto T_liq_loc = EOS_phase_liq.T_value_RhoP(rho_liq_loc, p_liq_loc);
                                               const auto T_g_loc   = EOS_phase_gas.T_value_RhoP(rho_g_loc, p_g_loc);
                                               const auto fac_T     = (chi_g_loc/T_g_loc + chi_liq_loc/T_liq_loc)/
                                                                      (Y_g_loc/T_g_loc + Y_liq_loc/T_liq_loc);
                                               if(dH > static_cast<Number>(0.0)) {
                                                 // Bound-preserving condition for m_l
                                                 dtau_ov_epsilon = lambda/(sigma*dH);
                                                 #ifdef DEBUG_RELAXATION
                                                   if(dtau_ov_epsilon < static_cast<Number>(0.0)) {
                                                     throw std::runtime_error("Negative time step found after relaxation of "
                                                                              "mass of large-scale liquid phase");
                                                   }
                                                 #endif

                                                 // Bound preserving for the velocity
                                                 auto dtau_ov_epsilon_tmp = lambda*mom_dot_vel/(fac_T*sigma*alpha_l_loc*dH*fac_Ru);
                                                                            // TODO: Add a check in case of zero volume fraction
                                                 dtau_ov_epsilon          = std::min(dtau_ov_epsilon, dtau_ov_epsilon_tmp);
                                                 #ifdef DEBUG_RELAXATION
                                                   if(dtau_ov_epsilon < static_cast<Number>(0.0)) {
                                                     throw std::runtime_error("Negative time step found after "
                                                                              "relaxation of velocity");
                                                   }
                                                 #endif

                                                 /*--- No specific condition to impose for the positivity of alpha_d
                                                       since alpha_d = alpha_l*m_d/m_l and m_d is increasing,
                                                       m_l has already been imposed positive and alpha_l is going to
                                                       be set with proper bounds later.
                                                       On the other hand, there is no a priori superior limit, apart from the
                                                       alpha_d_max which deactivates the mass transfer.
                                                       Hence, in the first iteration, one can potentially reach alpha_d > alpha_d_max
                                                       (likely unphysical, but not impossible...) ---*/
                                               }

                                               // Auxiliary partial derivatives
                                               // (for PRESSURE_EQUILIBRIUM F = delta_p, so dF = ddelta_p; they do not
                                               // contribute anyway since dH = 0 implies r_ml = B_liq = B_g = 0, i.e. R = 0)
                                               const auto ddelta_p_drhoz = -sigma*inv_rho_loc*
                                                                           std::cbrt(inv_rho_liq_loc*inv_rho_liq_loc)*
                                                                           rho_Gamma_jump;
                                               const auto dF_LS_drhoz    = alpha_l_loc*ddelta_p_drhoz;
                                               const auto dF_SS_drhoz    = static_cast<Number>(-2.0/3.0)*sigma*
                                                                           std::cbrt(inv_rho_liq_loc*inv_rho_liq_loc)
                                                                         + alpha_d_loc*ddelta_p_drhoz;
                                               const auto dF_drhoz       = (type_relaxation[cell] == PRESSURE_EQUILIBRIUM)?
                                                                           ddelta_p_drhoz : dF_LS_drhoz + dF_SS_drhoz;

                                               const auto ddelta_p_dmd = -m_g_loc*inv_alpha_g_loc*inv_alpha_g_loc*
                                                                         (c_g_loc*c_g_loc - p_g_loc/rho_g_loc*Gamma_g_loc)*
                                                                         inv_rho_liq_loc;
                                                                         // TODO: Add a check in case of zero volume fraction
                                               const auto dF_LS_dmd    = alpha_l_loc*ddelta_p_dmd;
                                               const auto dF_SS_dmd    = (delta_p + m_d_loc*ddelta_p_dmd)*inv_rho_liq_loc;
                                               const auto dF_dmd       = (type_relaxation[cell] == PRESSURE_EQUILIBRIUM)?
                                                                         ddelta_p_dmd : dF_LS_dmd + dF_SS_dmd;

                                               const auto ddelta_p_dml = (c_liq_loc*c_liq_loc - p_liq_loc/rho_liq_loc*Gamma_liq_loc)*
                                                                         inv_alpha_l_loc
                                                                       + m_g_loc*inv_alpha_g_loc*inv_alpha_g_loc*
                                                                         (c_g_loc*c_g_loc - p_g_loc/rho_g_loc*Gamma_g_loc)*
                                                                         alpha_d_loc*inv_m_l_loc
                                                                       + static_cast<Number>(2.0/3.0)*sigma*rho_z_loc*inv_rho_loc*
                                                                         std::cbrt(inv_rho_liq_loc*inv_rho_liq_loc)*inv_m_l_loc*
                                                                         rho_Gamma_jump;
                                                                         // TODO: Add a check in case of zero volume fraction
                                               const auto dF_LS_dml    = alpha_l_loc*ddelta_p_dml;
                                               const auto dF_SS_dml    = (m_d_loc*ddelta_p_dml -
                                                                          static_cast<Number>(1.0/3.0)*aux_SS)*inv_rho_liq_loc
                                                                       - F_SS*inv_m_l_loc;
                                                                         // TODO: Add a check in case of zero volume fraction
                                               const auto dF_dml       = (type_relaxation[cell] == PRESSURE_EQUILIBRIUM)?
                                                                         ddelta_p_dml : dF_LS_dml + dF_SS_dml;

                                               const auto fac_de_star    = (type_relaxation[cell] == PRESSURE_EQUILIBRIUM)?
                                                                           static_cast<Number>(1.0) : alpha_liq_loc;
                                               const auto dF_de_star_liq = fac_de_star*rho_liq_loc*Gamma_liq_loc;
                                               const auto dF_de_star_g   = -fac_de_star*rho_g_loc*Gamma_g_loc;

                                               // Auxiliary definitions
                                               const auto r_ml = -m_l_loc*sigma*dH;

                                               /* NOTE: Acoustic-impedance weighted interfacial pressure
                                                        (as in the instantaneous relaxation) seem to perform better,
                                                        but this choice is not consistent with the thermodynamic properties. */
                                               /*const auto Z_liq_loc = rho_liq_loc*c_liq_loc;
                                               const auto Z_g_loc   = rho_g_loc*c_g_loc;
                                               const auto pI_k_loc  = (Z_g_loc*p_liq_loc + Z_liq_loc*p_g_loc)/(Z_liq_loc + Z_g_loc);*/
                                               /* NOTE: This choice is made to ensure that the interfacial pressure
                                                        is consistent with the thermodynamic properties, but it seems to be
                                                        problematic (at least for sigma = 0).
                                                        With chi_k = Y_k one has pI ~ p_liq in gas-dominated cells,
                                                        so that residual liquid is relaxed with no dissipation.*/
                                               const auto pI_k_loc = chi_liq_loc*p_g_loc + chi_g_loc*p_liq_loc;
                                                                     /*NOTE: Valid assuming \partial_{\alpha_g}\chi_{g} = 0 */

                                               // Optionally freeze pI at its value at the first Newton iteration of the cell.
                                               // This is a rectangle rule for the interfacial work: the instantaneous relaxation
                                               // uses instead the trapezoidal rule with the average of pI_0 and p_star
                                               if(freeze_pI && Newton_iterations[cell] == 1) {
                                                 pI_0[cell] = pI_k_loc;
                                               }
                                               const auto pI_loc = freeze_pI ? pI_0[cell] : pI_k_loc;
                                               const auto B_liq  = -pI_loc*alpha_liq_loc*sigma*dH
                                                                 + Y_liq_loc*(fac_T*sigma*alpha_l_loc*dH*fac_Ru);
                                               const auto B_g    = pI_loc*alpha_liq_loc*sigma*dH
                                                                 + Y_g_loc*(fac_T*sigma*alpha_l_loc*dH*fac_Ru);
                                               const auto R      = r_ml*
                                                                   (dF_dml -
                                                                    dF_dmd -
                                                                    dF_drhoz*((static_cast<Number>(3.0)*Hmax/kappa)*
                                                                              std::cbrt(inv_rho_liq_loc)))
                                                                 + dF_de_star_liq*B_liq/m_liq_loc
                                                                 + dF_de_star_g*B_g/m_g_loc;
                                                                   /*NOTE: equivalent to dF_drhoz*(S_avg/m_avg)*((rho*z/Sigma))
                                                                           since S_avg/m_avg = 3Hmax/(kappa*rho_liq) and
                                                                           rho*z/Sigma = rho_liq^(2/3)*/

                                               const auto Y_l_loc = m_l_loc*inv_rho_loc;
                                               const auto A_liq   = std::isnan(H_loc)? -pI_loc*Y_liq_loc/Y_l_loc :
                                                                                       -chi_liq_loc*sigma*H_loc - pI_loc*Y_liq_loc/Y_l_loc;
                                               const auto A_g     = std::isnan(H_loc)? pI_loc*Y_liq_loc/Y_l_loc :
                                                                                       -chi_g_loc*sigma*H_loc + pI_loc*Y_liq_loc/Y_l_loc;
                                               const auto DF      = dF_dalpha_l
                                                                  + dF_de_star_liq*A_liq/m_liq_loc
                                                                  + dF_de_star_g*A_g/m_g_loc;

                                               // The bounds below assume 1 - dtau_ov_epsilon*DF > 0, not guaranteed a priori.
                                               // The polynomial bounds on alpha_l would already ensure it (up to the degenerate case),
                                               // but 1 - dtau_ov_epsilon*DF could be arbitrarily small:
                                               // we keep a margin to limit the amplification of F
                                               if(DF > static_cast<Number>(0.0)) {
                                                 dtau_ov_epsilon = std::min(dtau_ov_epsilon, lambda/DF);
                                               }

                                               // Bound-preserving condition for large-scale volume fraction
                                               // Upper bound
                                               const auto a             = R;
                                               auto b                   = F + lambda*(static_cast<Number>(1.0) - alpha_l_loc)*DF;
                                               auto D                   = b*b
                                                                        - static_cast<Number>(4.0)*a*
                                                                          (-lambda*(static_cast<Number>(1.0) - alpha_l_loc));
                                               auto dtau_ov_epsilon_tmp = std::numeric_limits<Number>::infinity();
                                               /*if(D > static_cast<Number>(0.0) &&
                                                  (a > static_cast<Number>(0.0) ||
                                                   (a < static_cast<Number>(0.0) &&
                                                    b > static_cast<Number>(0.0)))) {
                                                 dtau_ov_epsilon_tmp = static_cast<Number>(0.5)*(-b + std::sqrt(D))/a;
                                               }
                                               if(a == static_cast<Number>(0.0) &&
                                                  b > static_cast<Number>(0.0)) {
                                                 dtau_ov_epsilon_tmp = lambda*(static_cast<Number>(1.0) - alpha_l_loc)/b;
                                               }*/
                                               if(D > static_cast<Number>(0.0) &&
                                                  (a > static_cast<Number>(0.0) || b > static_cast<Number>(0.0))) {
                                                  if(b >= static_cast<Number>(0.0)) {
                                                    // Stable form, no cancellation, also a = 0 is included
                                                    dtau_ov_epsilon_tmp = static_cast<Number>(2.0)*
                                                                          (lambda*(static_cast<Number>(1.0) - alpha_l_loc))/
                                                                          (b + std::sqrt(D));
                                                  }
                                                  else {
                                                    // b < 0 (hence a > 0): -b + sqrt(D) is sum of two positive values
                                                    dtau_ov_epsilon_tmp = static_cast<Number>(0.5)*(-b + std::sqrt(D))/a;
                                                  }
                                               }
                                               dtau_ov_epsilon = std::min(dtau_ov_epsilon, dtau_ov_epsilon_tmp);
                                               // Lower bound
                                               dtau_ov_epsilon_tmp = std::numeric_limits<Number>::infinity();
                                               b                   = F - lambda*alpha_l_loc*DF;
                                               D                   = b*b
                                                                   - static_cast<Number>(4.0)*a*(lambda*alpha_l_loc);
                                               /*if(D > static_cast<Number>(0.0) &&
                                                  (a < static_cast<Number>(0.0) ||
                                                   (a > static_cast<Number>(0.0) &&
                                                    b < static_cast<Number>(0.0)))) {
                                                 dtau_ov_epsilon_tmp = static_cast<Number>(0.5)*(-b - std::sqrt(D))/a;
                                               }
                                               if(a == static_cast<Number>(0.0) &&
                                                  b < static_cast<Number>(0.0)) {
                                                 dtau_ov_epsilon_tmp = -lambda*alpha_l_loc/b;
                                               }*/
                                               if(D > static_cast<Number>(0.0) &&
                                                  (a < static_cast<Number>(0.0) || b < static_cast<Number>(0.0))) {
                                                  if(b <= static_cast<Number>(0.0)) {
                                                    // Stable form, no cancellation, also a = 0 is included
                                                    dtau_ov_epsilon_tmp = static_cast<Number>(2.0)*
                                                                          (lambda*alpha_l_loc)/
                                                                          (-b + std::sqrt(D));
                                                  }
                                                  else {
                                                    // b > 0 (hence a < 0): -b - sqrt(D) is sum of two negative values
                                                    dtau_ov_epsilon_tmp = static_cast<Number>(0.5)*(-b - std::sqrt(D))/a;
                                                  }
                                               }
                                               dtau_ov_epsilon = std::min(dtau_ov_epsilon, dtau_ov_epsilon_tmp);
                                               #ifdef DEBUG_RELAXATION
                                                 if(dtau_ov_epsilon < static_cast<Number>(0.0)) {
                                                   throw std::runtime_error("Negative time step found after relaxation of "
                                                                            "large-scale volume fraction");
                                                 }
                                               #endif

                                               // Thermodynamic admissibility for SG-EOS: check Q > 0, P_g > 0, P_liq > 0
                                               // at the candidate step and halve it otherwise (Q(0) > 0, P_k(0) > 0).
                                               // delta = 1 - dtau*DF is passed separately to handle also dtau -> inf
                                               const auto z_loc  = rho_z_loc*inv_rho_loc;
                                               const auto fac_z  = inv_rho_loc*(static_cast<Number>(3.0)*Hmax/kappa)*
                                                                   std::cbrt(inv_rho_liq_loc);
                                               auto thermo_admissible = [&](const Number dtau, const Number delta) {
                                                 const auto m_l_new       = m_l_loc + dtau*r_ml;
                                                 const auto delta_F_new   = F + dtau*R; /*--- delta*F^{k+1} ---*/
                                                 const auto delta_alpha_l = delta*alpha_l_loc + dtau*delta_F_new;
                                                                            /*--- delta*alpha_l^{k+1} ---*/
                                                 const auto z_new         = z_loc - dtau*r_ml*fac_z;
                                                 const auto Q             = delta*m_l_new - m_liq_loc*delta_alpha_l;

                                                 // P_k for phase k
                                                 auto P = [&](const auto& EOS, const Number e_k, const Number m_k,
                                                              const Number chi_ov_Y_k, const Number A_k, const Number B_k,
                                                              const Number delta_m_l_ov_rho_k) {
                                                   return delta*m_l_new*
                                                          (e_k - EOS.get_q_infty() +
                                                           sigma*chi_ov_Y_k*std::cbrt(inv_rho_liq_loc*inv_rho_liq_loc)*
                                                           (z_loc - z_new/static_cast<Number>(3.0)) +
                                                           dtau*B_k/m_k)
                                                        + m_l_new*dtau*A_k*delta_F_new/m_k
                                                        - static_cast<Number>(2.0/3.0)*sigma*chi_ov_Y_k*
                                                          std::cbrt(rho_liq_loc)*z_new*delta_alpha_l
                                                        - EOS.get_pi_infty()*delta_m_l_ov_rho_k;
                                                 };

                                                 return Q > static_cast<Number>(0.0) &&
                                                        P(EOS_phase_liq, e_liq_loc, m_liq_loc, chi_liq_loc/Y_liq_loc,
                                                          A_liq, B_liq, delta_alpha_l) > static_cast<Number>(0.0) &&
                                                        P(EOS_phase_gas, e_g_loc, m_g_loc, chi_g_loc/Y_g_loc,
                                                          A_g, B_g, Q/m_g_loc) > static_cast<Number>(0.0);
                                               };
                                               if(std::isinf(dtau_ov_epsilon) &&
                                                  !thermo_admissible(static_cast<Number>(1.0), -DF)) {
                                                 // Here dH = 0 (no mass transfer) and DF < 0: Q and P_k are
                                                 // homogeneous of degree one in (dtau, delta), so (1, -DF) checks
                                                 // the Newton step. If not admissible, restart from delta = 2
                                                 dtau_ov_epsilon = static_cast<Number>(-1.0)/DF;
                                               }
                                               if(!std::isinf(dtau_ov_epsilon)) {
                                                 std::size_t n_halving = 0;
                                                 while(!thermo_admissible(dtau_ov_epsilon,
                                                                          static_cast<Number>(1.0) - dtau_ov_epsilon*DF)) {
                                                   if(++n_halving > 100) {
                                                     throw std::runtime_error("No thermodynamically admissible time step "
                                                                              "found inside Newton step");
                                                   }
                                                   dtau_ov_epsilon *= static_cast<Number>(0.5);
                                                 }
                                               }

                                               // Compute the effective variation of the variables
                                               if(std::isinf(dtau_ov_epsilon)) {
                                                 // If we are in this branch we do not have mass transfer
                                                 // and we have no other restrictions on the bounds of large scale volume fraction
                                                 dalpha_l_loc = -F/DF;
                                               }
                                               else {
                                                 const auto dm_l = dtau_ov_epsilon*r_ml;

                                                 #ifdef DEBUG_RELAXATION
                                                   if(dm_l > static_cast<Number>(0.0)) {
                                                     throw std::runtime_error("Negative sign of mass transfer inside Newton step");
                                                   }
                                                 #endif
                                                 result(Ml_INDEX) += dm_l;
                                                 #ifdef DEBUG_RELAXATION
                                                   if(result(Ml_INDEX) < static_cast<Number>(0.0)) {
                                                     // I should never get here. Added only for the sake of safety!!
                                                     throw std::runtime_error("Negative mass of large-scale "
                                                                              "liquid phase inside Newton step");
                                                   }
                                                 #endif

                                                 result(Md_INDEX) -= dm_l;
                                                 #ifdef DEBUG_RELAXATION
                                                   if(result(Md_INDEX) < static_cast<Number>(0.0)) {
                                                     // I should never get here. Added only for the sake of safety!!
                                                     throw std::runtime_error("Negative mass of small-scale "
                                                                              "liquid phase inside Newton step");
                                                   }
                                                 #endif

                                                 const auto R_Sigma_d = -dm_l*((static_cast<Number>(3.0)*Hmax/kappa)*
                                                                               inv_rho_liq_loc);
                                                 result(RHO_Z_INDEX) += std::cbrt(rho_liq_loc*rho_liq_loc)*R_Sigma_d;

                                                 /*--- u/u^{2} = rho*u/(rho*(u^{2})) = (rho/(rho*u)^{2})*(rho*u) ---*/
                                                 const auto drho_fac_Ru = dtau_ov_epsilon*
                                                                          (fac_T*sigma*alpha_l_loc*dH*fac_Ru)*rho_loc/mom_squared;
                                                 for(std::size_t d = 0; d < Field::dim; ++d) {
                                                   result(RHO_U_INDEX + d) -= drho_fac_Ru*result(RHO_U_INDEX + d);
                                                 }

                                                 dalpha_l_loc = dtau_ov_epsilon/(static_cast<Number>(1.0) - dtau_ov_epsilon*DF)*
                                                                (F + dtau_ov_epsilon*R);
                                               }

                                               #ifdef DEBUG_RELAXATION
                                                 if(alpha_l_loc + dalpha_l_loc < static_cast<Number>(0.0) ||
                                                    alpha_l_loc + dalpha_l_loc > static_cast<Number>(1.0)) {
                                                   // I should never get here. Added only for the sake of safety!!
                                                   throw std::runtime_error("Bounds exceeding value for large-scale volume fraction "
                                                                            "inside Newton step");
                                                 }
                                               #endif
                                               alpha_l_loc += dalpha_l_loc;
                                               alpha_l[cell]  = alpha_l_loc;
                                               dalpha_l[cell] = dalpha_l_loc;
                                               result(ALPHA_l_INDEX) = alpha_l_loc;
                                               if(std::isinf(dtau_ov_epsilon)) {
                                                 result(Mliq_Eliq_INDEX) += A_liq*dalpha_l_loc;

                                                 result(Mg_Eg_INDEX) += A_g*dalpha_l_loc;
                                               }
                                               else {
                                                 result(Mliq_Eliq_INDEX) += (A_liq*dalpha_l_loc + dtau_ov_epsilon*B_liq);

                                                 result(Mg_Eg_INDEX) += (A_g*dalpha_l_loc + dtau_ov_epsilon*B_g);
                                               }
                                             }

                                             // Check if resulting state is admissible
                                             const auto m_d_loc_res       = result(Md_INDEX);
                                             const auto m_l_loc_res       = result(Ml_INDEX);
                                             const auto m_liq_loc_res     = m_l_loc_res + m_d_loc_res;
                                             const auto m_g_loc_res       = result(Mg_INDEX);
                                             const auto rho_loc_res       = m_liq_loc_res + m_g_loc_res;
                                             /* NOTE: This is in principle equal to m_liq_loc computed at the beginning
                                                      of the relaxation but, for the sake of coherence, and to avoid any
                                                      round-off effect we take the final value */
                                             const auto alpha_d_loc_res   = alpha_l_loc*m_d_loc_res/m_l_loc_res;
                                                                            // TODO: Add a check in case of zero volume fraction
                                             const auto alpha_liq_loc_res = alpha_l_loc + alpha_d_loc_res;
                                             const auto rho_liq_loc_res   = m_liq_loc_res/alpha_liq_loc_res;
                                             const auto Sigma_d_loc_res   = result(RHO_Z_INDEX)/
                                                                            std::cbrt(rho_liq_loc_res*rho_liq_loc_res);
                                             const auto Y_liq_loc_res     = m_liq_loc_res/rho_loc_res;
                                             const auto chi_liq_loc_res   = Y_liq_loc_res;
                                             const auto e_liq_loc_res     = result(Mliq_Eliq_INDEX)/m_liq_loc_res
                                                                          - sigma*(chi_liq_loc_res/(rho_loc_res*Y_liq_loc_res))*
                                                                            Sigma_d_loc_res;
                                             const auto p_liq_loc_res     = EOS_phase_liq.pres_value_Rhoe(rho_liq_loc_res,
                                                                                                          e_liq_loc_res);
                                             const auto c_liq_loc_res     = EOS_phase_liq.c_value_RhoP(rho_liq_loc_res,
                                                                                                       p_liq_loc_res);
                                             if(std::isnan(c_liq_loc_res)) {
                                                std::cerr << "p_liq_loc_res = " << p_liq_loc_res << std::endl;
                                                throw std::runtime_error("Non admissible liquid pressure inside Newton step");
                                             }

                                             const auto alpha_g_loc_res = static_cast<Number>(1.0) - alpha_liq_loc_res;
                                             const auto rho_g_loc_res   = m_g_loc_res/alpha_g_loc_res;
                                             const auto Y_g_loc_res     = static_cast<Number>(1.0) - Y_liq_loc_res;
                                             const auto chi_g_loc_res   = Y_g_loc_res;
                                             const auto e_g_loc_res     = result(Mg_Eg_INDEX)/m_g_loc_res
                                                                        - sigma*(chi_g_loc_res/(rho_loc_res*Y_g_loc_res))*
                                                                          Sigma_d_loc_res;
                                             const auto p_g_loc_res     = EOS_phase_gas.pres_value_Rhoe(rho_g_loc_res, e_g_loc_res);
                                             const auto c_g_loc_res     = EOS_phase_gas.c_value_RhoP(rho_g_loc_res, p_g_loc_res);
                                             if(std::isnan(c_g_loc_res)) {
                                                std::cerr << "p_g_loc_res = " << p_g_loc_res << std::endl;
                                                throw std::runtime_error("Non admissible gas pressure inside Newton step");
                                             }
                                           });

    return relaxation_step;
  }

  // Implement the instantaneous relaxation for the pressure (linearization method Pelanti based),
  // with phase 1 = liquid (large + small scale) and phase 2 = gas.
  // Since sigma = 0 when using this routine, the internal-energy formulation stores m_liq*e_liq and m_g*e_g.
  // The ratio alpha_d/alpha_l = m_d/m_l is preserved, so that alpha_l = alpha_liq*m_l/m_liq.
  //
  template<class Field>
  template<class Field_Scalar>
  auto RelaxationOperator<Field>::make_instantaneous_pressure_relaxation(Field_Scalar& alpha_l) {
    auto relaxation_step = samurai::make_cell_based_scheme<typename RelaxationOperator::cfg>();
    relaxation_step.set_name("Instantaneous pressure relaxation");
    relaxation_step.set_scheme_function([&](samurai::SchemeValue<cfg>& result, const auto& cell, const auto& field)
                                           {
                                             const auto local_field = field[cell];
                                             result = field[cell];

                                             // Pre-fetch variables used multiple times to exploit possible vectorization
                                             const auto m_l_loc   = local_field(Ml_INDEX);
                                             const auto m_g_loc   = local_field(Mg_INDEX);
                                             const auto m_d_loc   = local_field(Md_INDEX);
                                             const auto m_liq_loc = m_l_loc + m_d_loc;

                                             const auto alpha_l_loc = local_field(ALPHA_l_INDEX);
                                             auto alpha_liq_loc     = alpha_l_loc*m_liq_loc/m_l_loc;
                                                                      // TODO: Add a check in case of zero volume fraction
                                             const auto alpha_g_loc = static_cast<Number>(1.0) - alpha_liq_loc;

                                             // Compute initial values of the pressure (i.e. after hyperbolic operator)
                                             const auto e_liq_0_loc = local_field(Mliq_Eliq_INDEX)/m_liq_loc;
                                             const auto e_g_0_loc   = local_field(Mg_Eg_INDEX)/m_g_loc;

                                             const auto rho_liq_loc = m_liq_loc/alpha_liq_loc;
                                                                      // TODO: Add a check in case of zero volume fraction
                                             const auto p_liq_loc   = EOS_phase_liq.pres_value_Rhoe(rho_liq_loc, e_liq_0_loc);

                                             const auto rho_g_loc = m_g_loc/alpha_g_loc;
                                                                    // TODO: Add a check in case of zero volume fraction
                                             const auto p_g_loc   = EOS_phase_gas.pres_value_Rhoe(rho_g_loc, e_g_0_loc);

                                             // Compute the pressure equilibrium with the linearization method (Pelanti)
                                             const auto gamma_liq    = EOS_phase_liq.get_gamma();
                                             const auto gamma_g      = EOS_phase_gas.get_gamma();
                                             const auto pi_infty_liq = EOS_phase_liq.get_pi_infty();
                                             const auto pi_infty_g   = EOS_phase_gas.get_pi_infty();

                                             const auto a     = static_cast<Number>(1.0)
                                                              + gamma_g*alpha_liq_loc
                                                              + gamma_liq*alpha_g_loc;
                                             // Acoustic-impedance weighted interfacial pressure
                                             /*const auto c_liq_loc = EOS_phase_liq.c_value_RhoP(rho_liq_loc, p_liq_loc);
                                             const auto c_g_loc   = EOS_phase_gas.c_value_RhoP(rho_g_loc, p_g_loc);
                                             const auto Z_liq     = rho_liq_loc*c_liq_loc;
                                             const auto Z_g       = rho_g_loc*c_g_loc;
                                             const auto pI_0      = (Z_g*p_liq_loc + Z_liq*p_g_loc)/(Z_liq + Z_g);*/
                                             /* NOTE: Same interfacial pressure as in the Newton step (chi_k = Y_k) */
                                             const auto Y_liq_loc = m_liq_loc/(m_liq_loc + m_g_loc);
                                             const auto pI_0      = Y_liq_loc*p_g_loc
                                                                  + (static_cast<Number>(1.0) - Y_liq_loc)*p_liq_loc;
                                             const auto C_liq = static_cast<Number>(2.0)*gamma_liq*pi_infty_liq
                                                              + (gamma_liq - static_cast<Number>(1.0))*pI_0;
                                             const auto C_g   = static_cast<Number>(2.0)*gamma_g*pi_infty_g
                                                              + (gamma_g - static_cast<Number>(1.0))*pI_0;
                                             const auto b     = C_liq*alpha_g_loc + C_g*alpha_liq_loc
                                                              - (static_cast<Number>(1.0) + gamma_g)*alpha_liq_loc*p_liq_loc
                                                              - (static_cast<Number>(1.0) + gamma_liq)*alpha_g_loc*p_g_loc;
                                             const auto d     = -(C_g*alpha_liq_loc*p_liq_loc + C_liq*alpha_g_loc*p_g_loc);

                                             const auto p_star = (-b + std::sqrt(b*b - static_cast<Number>(4.0)*a*d))/
                                                                 (static_cast<Number>(2.0)*a);

                                             // Update the volume fraction using the computed pressure
                                             alpha_liq_loc *= ((gamma_liq - static_cast<Number>(1.0))*p_star +
                                                               static_cast<Number>(2.0)*p_liq_loc + C_liq)/
                                                              ((gamma_liq + static_cast<Number>(1.0))*p_star + C_liq);
                                             #ifdef DEBUG_RELAXATION
                                               if(std::isnan(p_star) ||
                                                  alpha_liq_loc < static_cast<Number>(0.0) ||
                                                  alpha_liq_loc > static_cast<Number>(1.0)) {
                                                 std::cerr << "p_liq = " << p_liq_loc << ", p_g = " << p_g_loc
                                                           << ", p_star = " << p_star << ", alpha_liq = " << alpha_liq_loc << std::endl;
                                                 throw std::runtime_error("Non admissible state in the instantaneous pressure relaxation");
                                               }
                                             #endif
                                             const auto alpha_l_new_loc = alpha_liq_loc*m_l_loc/m_liq_loc;
                                             alpha_l[cell]              = alpha_l_new_loc;
                                             result(ALPHA_l_INDEX)      = alpha_l_new_loc;

                                             // Update the internal energy of both phases
                                             result(Mliq_Eliq_INDEX) = m_liq_loc*
                                                                       EOS_phase_liq.e_value_RhoP(m_liq_loc/alpha_liq_loc, p_star);
                                                                       // TODO: Add a check in case of zero volume fraction
                                             result(Mg_Eg_INDEX)     = m_g_loc*
                                                                       EOS_phase_gas.e_value_RhoP(m_g_loc/
                                                                                                  (static_cast<Number>(1.0) - alpha_liq_loc),
                                                                                                  p_star);
                                                                       // TODO: Add a check in case of zero volume fraction
                                           });

    return relaxation_step;
  }

} // end of namespace