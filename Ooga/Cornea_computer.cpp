#include <math.h>
#include <stdlib.h>
#include "Cornea_computer.h"
//#include "trackerSettings.h"
#include <stdio.h>
#include <iostream>

namespace gt {

    static const unsigned int MAX_ITER = 1000;
    static const double PRECISION      = 0.000000000001;


    //#define POW2(X) ((X) * (X))
    static double POW2(double x) {return x*x;}


    // when using the solver
    static int my_fdf(const gsl_vector *x, void *param, gsl_vector *ret_vec, gsl_matrix *ret_mat);
    static int my_df(const gsl_vector *x, void *param, gsl_matrix *ret);
    static int my_f(const gsl_vector *x, void *param, gsl_vector *ret);


    inline int pairsOfTwo(int a) {

        --a;

        int res = 0;
        for(; a > 0; --a) {
            res += a;
        }

        return res;
    }


    void Cornea::create(const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &led_pos,		// LED locations
                        const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &glint_pos) {	// corresponding glint locations

        const size_t sz = led_pos.size();

        // resize the data vector
        data.resize(sz);


        for(size_t i = 0; i < sz; ++i) {

            DATA_FOR_CORNEA_COMPUTATION &cur_data = data[i];

            const Eigen::Vector3d &cur_led   = led_pos[i];
            const Eigen::Vector3d &cur_glint = glint_pos[i];


            // create the rotation matrices
            Cornea::createRotationMatrix(cur_led, cur_glint, cur_data.R);

            // calculate the inverse rotation matrices
            //cur_data.R.computeInverse(&(cur_data.R_inv));
            cur_data.R_inv = cur_data.R.inverse();


            // lets calculate l_aux1 and l_aux2, in the auxiliary coordinate system
            cur_data.l_aux = (cur_data.R_inv * cur_led).norm();

            // calculate alpha_aux
            const Eigen::Vector3d g_aux_n = (cur_data.R_inv * cur_glint).normalized();

            /*
             * The correct way to compute the angle between 2 vectors is
             *
             *     cur_data.alpha_aux = acos(g_aux_n.dot(q_aux1.normalized()));
             *
             * ...however, we know that q_aux* vectors are on the x-axis => q_aux_n* = (1, 0, 0)
             */
            cur_data.alpha_aux = acos(g_aux_n(0));

	    // std::cout << "cur_data.alpha_aux: " << cur_data.alpha_aux << std::endl;
	    // std::cout << "cur_data.l_aux: " << cur_data.l_aux << std::endl;

	    
	    //std::cout << "g_aux: "  << g_aux_n(0) << "  " << g_aux_n(1) << "  " << g_aux_n(2) << std::endl;

        }

    }


    /*
     * Creates a rotation matrix which describes how a point in an auxiliary
     * coordinate system, whose x axis is desbibed by vec_along_x_axis and has
     * a point on its xz-plane vec_on_xz_plane, rotates into the real coordinate
     * system.
     */
    void Cornea::createRotationMatrix(const Eigen::Vector3d &vec_along_x_axis,
                                      const Eigen::Vector3d &vec_on_xz_plane,
                                      Eigen::Matrix3d &R) {

        // normalise pw
        Eigen::Vector3d vec_on_xz_plane_n = vec_on_xz_plane.normalized();

        // define helper variables x, y and z
        // x
        Eigen::Vector3d xn = vec_along_x_axis.normalized();

        // y
        Eigen::Vector3d tmp = vec_on_xz_plane_n.cross(xn);
        Eigen::Vector3d yn = tmp.normalized();

        // z
        tmp = xn.cross(yn);
        Eigen::Vector3d zn = tmp.normalized();

        // create the rotation matrix
        R.col(0) << xn(0), xn(1), xn(2);
        R.col(1) << yn(0), yn(1), yn(2);
        R.col(2) << zn(0), zn(1), zn(2);

    }

  int Cornea::testiF() {
    std::cout << "jepa  " << std::endl;
	return 0; //must return value
  }

    int Cornea::computeCentre(const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &led_pos, // LED locations
                              const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &glint_pos,
                              std::vector<double> &gx_guesses,
                              Eigen::Vector3d &centre,
                              double &err) {

        // initialise the cornea tracker
        create(led_pos, glint_pos);

        /*
         * Check out usage and more info about GSL:
         * http://www.csse.uwa.edu.au/programming/gsl-1.0/gsl-ref_35.html
         */

        const size_t n = 3 * pairsOfTwo(data.size());	// number of functions
        const size_t p = gx_guesses.size();				// number of parameters

        // initial guesses
        gsl_vector_const_view x = gsl_vector_const_view_array(gx_guesses.data(), p);

        gsl_multifit_function_fdf f;
        f.f			= &my_f;	// function
        f.df		= &my_df;	// derivative
        f.fdf		= &my_fdf;	// both
        f.n			= n;		// number of functions
        f.p			= p;		// number of parameters
        f.params	= this;		// additional parameter

        const gsl_multifit_fdfsolver_type *T = gsl_multifit_fdfsolver_lmsder;
        gsl_multifit_fdfsolver *solver = gsl_multifit_fdfsolver_alloc(T, n, p);
        gsl_multifit_fdfsolver_set(solver, &f, &x.vector);


        int status;
        unsigned int iter = 0;

        do {
            iter++;
            status = gsl_multifit_fdfsolver_iterate(solver);

            if(status) {
                break;
            }

            status = gsl_multifit_test_delta(solver->dx, solver->x, PRECISION, PRECISION);

        }
        while(status == GSL_CONTINUE && iter < MAX_ITER);

        if(iter == MAX_ITER) {
            printf("Cornea::computeCentre(): iter = MAX_ITER\n");
        }


        gsl_matrix *covar = gsl_matrix_alloc(p, p);
        gsl_multifit_covar(solver->J, 0.0, covar);

        // for(int row = 0; row < p; ++row) {
        //     for(int col = 0; col < p; ++col) {
        //         printf("%.2f ", covar->data[row * p + col]);
        //     }
        //     printf("\n");
        // }
        // printf("*****************************\n");

        /***********************************************************************
         * Compute the fit error
         **********************************************************************/

        err = 0;
        for(size_t i = 0; i < p; i++) {
            err += gsl_matrix_get(covar, i, i);
        }
        err = std::sqrt(err);


        Eigen::Vector3d cw(0.0, 0.0, 0.0);

        // cornea sphere radius
        //const double RHO = trackerSettings.RHO;
	const double RHO = 0.0077;

        // remove this
        double dMaxX = -10.0;

        for(size_t i = 0; i < data.size(); ++i) {

            const DATA_FOR_CORNEA_COMPUTATION &cur_data = data[i];

            const double gx_guess = gsl_vector_get(solver->x, i);

            const double B_aux = atan2(gx_guess * tan(cur_data.alpha_aux), (cur_data.l_aux - gx_guess));

            // calculate the corneal sphere centers in the auxiliary coordinate systems
            const Eigen::Vector3d c_aux(gx_guess - RHO * sin((cur_data.alpha_aux - B_aux) / 2.),
                                        0.,
                                        gx_guess * tan(cur_data.alpha_aux) + RHO * cos((cur_data.alpha_aux - B_aux) / 2.));

            const Eigen::Vector3d tmp = cur_data.R * c_aux;

            cw(0) += tmp(0);
            cw(1) += tmp(1);
            cw(2) += tmp(2);

            if(tmp(0) > dMaxX) {
                dMaxX = tmp(0);
            }

            //        printf("%i: centre (mm): %.2f %.2f %.2f\n", (int)i, 1000.0*tmp(0), 1000.0*tmp(1), 1000.0*tmp(2));
	    //std::cout << gx_guess << " ::: " << c_aux[0] << " ::: " << cur_data.l_aux << " ::: "  << cur_data.alpha_aux << " ::: " << B_aux << std::endl;

        }

        const double nof_samples = (double)data.size();
        centre << cw(0) / nof_samples, cw(1) / nof_samples, cw(2) / nof_samples;

	// miikan juttu:
        for(int i = 0; i < data.size(); ++i)
	  gx_guesses[i] = gsl_vector_get(solver->x, i);

        // printf("Avg: %.2f %.2f %.2f\n", 1000.0*centre(0), 1000.0*centre(1), 1000.0*centre(2));
        // printf("*********************************\n");

        gsl_multifit_fdfsolver_free(solver);
        gsl_matrix_free(covar);

        return (int)iter;

    }


    // see "computation_of_the_gaze.odt"
    void Cornea::createF(const gsl_vector *x, gsl_vector *F) const {

        // number of LEDs
        const int nLEDs = data.size();

        // index for F
        int index_f = 0;

        //const double RHO = trackerSettings.RHO;
	const double RHO = 0.0077;

        /*
         * For each LED pair, there are 3 functions
         */
        for(int i = 0; i < nLEDs-1; ++i) {

            const double gx_guess1 = gsl_vector_get(x, i);

            const DATA_FOR_CORNEA_COMPUTATION &data1 = data[i];

            const double tan_a1    = tan(data1.alpha_aux);
            const double atan_res1 = atan2(gx_guess1 * tan_a1, data1.l_aux - gx_guess1);
            const double cos_res1  = cos((data1.alpha_aux - atan_res1) * 0.5); // 0.5 = 1. / 2. => a / 2 = a*0.5
            const double sin_res1  = sin((data1.alpha_aux - atan_res1) * 0.5);

            const double A1 = data1.R(0, 0) * (gx_guess1 - RHO * sin_res1);
            const double B1 = data1.R(0, 2) * (gx_guess1 * tan_a1 + RHO * cos_res1);

            const double A2 = data1.R(1, 0) * (gx_guess1 - RHO * sin_res1);
            const double B2 = data1.R(1, 2) * (gx_guess1 * tan_a1 + RHO * cos_res1);

            const double A3 = data1.R(2, 0) * (gx_guess1 - RHO * sin_res1);
            const double B3 = data1.R(2, 2) * (gx_guess1 * tan_a1 + RHO * cos_res1);

            for(int j = i + 1; j < nLEDs; ++j) {

                const double gx_guess2 = gsl_vector_get(x, j);

                const DATA_FOR_CORNEA_COMPUTATION &data2 = data[j];

                const double tan_a2    = tan(data2.alpha_aux);
                const double atan_res2 = atan2(gx_guess2 * tan_a2, data2.l_aux - gx_guess2);
                const double cos_res2  = cos((data2.alpha_aux - atan_res2) * 0.5);
                const double sin_res2  = sin((data2.alpha_aux - atan_res2) * 0.5);

                const double C1 = data2.R(0, 0) * (gx_guess2 - RHO * sin_res2);
                const double D1 = data2.R(0, 2) * (gx_guess2 * tan_a2 + RHO * cos_res2);

                const double C2 = data2.R(1, 0) * (gx_guess2 - RHO * sin_res2);
                const double D2 = data2.R(1, 2) * (gx_guess2 * tan_a2 + RHO * cos_res2);

                const double C3 = data2.R(2, 0) * (gx_guess2 - RHO * sin_res2);
                const double D3 = data2.R(2, 2) * (gx_guess2 * tan_a2 + RHO * cos_res2);

                // assign the values
                gsl_vector_set(F, index_f,     A1 + B1 -C1 -D1);
                gsl_vector_set(F, index_f + 1, A2 + B2 -C2 -D2);
                gsl_vector_set(F, index_f + 2, A3 + B3 -C3 -D3);

                index_f += 3;

            }

        }

    }


    void Cornea::createJacobian(const gsl_vector *gx,
                                gsl_matrix *J) const { // x x y Jacobian matrix

      //const double RHO = trackerSettings.RHO;
      const double RHO = 0.0077;

        int ind_J = 0;

        const int nLEDs = data.size();
        const size_t rows     = 3*pairsOfTwo(nLEDs);

        // zero the Jacobian matrix
        for(size_t row = 0; row < rows; ++row) {
            for(size_t col = 0; col < nLEDs; ++col) {
                gsl_matrix_set(J, row, col, 0.0);
            }
        }


        for(size_t x = 0; x < nLEDs - 1; ++x) {

            const DATA_FOR_CORNEA_COMPUTATION &data1 = data[x];
            const double gx_guess1 = gsl_vector_get(gx, x);
            const double tan_a1    = tan(data1.alpha_aux);
            const double a1        = gx_guess1 * tan_a1 / (data1.l_aux - gx_guess1);
            const double atan_res1 = atan2(gx_guess1 * tan_a1, data1.l_aux - gx_guess1);

            double dA_plus_dB[3];

            for(int i = 0; i < 3; ++i) {

                /*******************************************************
                 * Get the gxi component
                 *******************************************************/

                const double rx0 = data1.R(i, 0);

                // d/dgxi A(gxi)
                const double dA =
                    rx0 + rx0 * RHO * cos((data1.alpha_aux - atan_res1) / 2.) *
                    (tan_a1 * (1.0 / (data1.l_aux - gx_guess1)) +
                     (1.0 / POW2(data1.l_aux - gx_guess1)) * gx_guess1 * tan_a1) / (2. * (1. + POW2(a1)));

                const double rx1 = data1.R(i, 2);

                // d/dgxi B(gxi)
                const double dB = 
                    rx1 * tan_a1 + rx1 * RHO * sin((data1.alpha_aux - atan_res1) / 2.) *
                    (tan_a1 * (1.0 / (data1.l_aux - gx_guess1)) +
                     (1.0 / POW2(data1.l_aux - gx_guess1)) * gx_guess1 * tan_a1) / (2. * (1. + POW2(a1)));

                dA_plus_dB[i] = dA + dB;

            }


            for(size_t y = x + 1; y < nLEDs; ++y) {

                const DATA_FOR_CORNEA_COMPUTATION &data2 = data[y];
                const double gx_guess2 = gsl_vector_get(gx, y);
                const double tan_a2    = tan(data2.alpha_aux);
                const double a2        = gx_guess2 * tan_a2 / (data2.l_aux - gx_guess2);
                const double atan_res2 = atan2(gx_guess2 * tan_a2, data2.l_aux - gx_guess2);

                for(int i = 0; i < 3; ++i) {

                    /******************************************************************************************
                     * Get the gxj component
                     ******************************************************************************************/

                    const double rx0 = data2.R(i, 0);

                    // d/dgxj C(gxj)
                    const double dC =
                        rx0 + rx0 * RHO * cos((data2.alpha_aux - atan_res2) / 2.) *
                        (tan_a2 * (1.0 / (data2.l_aux - gx_guess2)) +
                         (1.0 / POW2(data2.l_aux - gx_guess2)) * gx_guess2 * tan_a2) / (2. * (1. + POW2(a2)));

                    const double rx1 = data2.R(i, 2);

                    // d/dgxj D(gxj)
                    const double dD =
                        rx1 * tan_a2 + rx1 * RHO * sin((data2.alpha_aux - atan_res2) / 2.) *
                        (tan_a2 * (1.0 / (data2.l_aux - gx_guess2)) +
                         (1.0 / POW2(data2.l_aux - gx_guess2)) * gx_guess2 * tan_a2) / (2. * (1. + POW2(a2)));

                    /*
                     * Set only 2 values per row, since we consider LED pairs,
                     * i.e. no other LED affects the jacobian in this row than
                     * one of the the LEDs in the pair in question.
                     */
                    gsl_matrix_set(J, ind_J, x, dA_plus_dB[i]);
                    gsl_matrix_set(J, ind_J, y, -dC - dD);

                    ++ind_J;

                }

            }

        }

    }



    /*************************************************************
     * GLS callbacks
     *************************************************************/

    int my_f(const gsl_vector *x, void *param, gsl_vector *F) {

        // get a pointer to the Cornea object
        const Cornea *c = (const Cornea *)param;

        c->createF(x, F);

        return GSL_SUCCESS;
    }


    int my_df(const gsl_vector *x, void *param, gsl_matrix *J) {

        // get a pointer to the Cornea object
        const Cornea *c = (const Cornea *)param;

        c->createJacobian(x, J);


        return GSL_SUCCESS;
    }


    /* Compute both F and dF */
    int my_fdf(const gsl_vector *x, void *param, gsl_vector *F, gsl_matrix *J) {

        // get a pointer to the Cornea object
        const Cornea *c = (const Cornea *)param;

        /****************************************************
         * F
         ****************************************************/
        c->createF(x, F);


        /****************************************************
         * dF
         ****************************************************/
        c->createJacobian(x, J);


        return GSL_SUCCESS;
    }


}    // end of "namespace gt"

