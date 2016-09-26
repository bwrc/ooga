/*
 * The computations are based on
 * A Single Camera Eye-Gaze Tracking System with Free Head Motion
 */


#ifndef CORNEA_COMPUTER_H
#define CORNEA_COMPUTER_H


#include <Eigen/Core>
#include <Eigen/QR>
#include <Eigen/LU>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

 //this here for gsl compatibility (__iob_func & file definitions changed from VS2013 -> VS2015)
 //see http://stackoverflow.com/questions/30412951/unresolved-external-symbol-imp-fprintf-and-imp-iob-func-sdl2
 //and https://msdn.microsoft.com/en-us/library/bb531344.aspx

//FILE _iob[] = { *stdin, *stdout, *stderr };

/*extern "C" FILE * __cdecl __iob_func(void)
{
	return _iob;
}

//extern "C" { FILE ___iob_func[3] = { *stdin,*stdout,*stderr }; }
extern "C" FILE* __cdecl __iob_func()
{
	struct _iobuf_VS2012 { // ...\Microsoft Visual Studio 11.0\VC\include\stdio.h #56
		char *_ptr;
		int   _cnt;
		char *_base;
		int   _flag;
		int   _file;
		int   _charbuf;
		int   _bufsiz;
		char *_tmpfname;
	};
	// VS2015 has only FILE = struct {void*}

	int const count = sizeof(_iobuf_VS2012) / sizeof(FILE);

	//// stdout
	//return (FILE*)(&(__acrt_iob_func(1)->_Placeholder) - count);

	// stderr
	return (FILE*)(&(__acrt_iob_func(2)->_Placeholder) - 2 * count);
}
*/

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_test.h>
#include <gsl/gsl_multiroots.h>
#include <gsl/gsl_multimin.h>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_multifit_nlin.h>

#include <vector>


namespace ooga {

    class DATA_FOR_CORNEA_COMPUTATION {

    public:

        // http://eigen.tuxfamily.org/dox-devel/group__TopicStructHavingEigenMembers.html
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        DATA_FOR_CORNEA_COMPUTATION() {
  
            l_aux		= 0.0;
            alpha_aux	= 0.0;

        }

        double l_aux;
        double alpha_aux;
        Eigen::Matrix3d R;
        Eigen::Matrix3d R_inv;

    };


    /*
     * This class solves the cornea centre in the eye camera coordinate
     * system. The reader must read "computation_of_the_gaze.odt" in order
     * to understand what is happening. Also the naming convention follows
     * that used in this documentation.
     */
    class Cornea {

    public:

        Cornea() {}

	int testiF();

        int computeCentre(const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &led_pos, // LED locations
                          const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &glint_pos,
                          std::vector<double> &gx_guesses,
                          Eigen::Vector3d &centre,
                          double &err);

        /*
         * The number of LEDs determines the number of equations, i.e. the
         * length of F vector. The LED pair count can be obtained by:
         *
         *     p = {i=1, 2, .., N-1} sum(N-i)
         *     where N is the LED count
         *
         * Each LED pair introduces three equations and a single unknown.
         * So for example if N
         * is 4, there will be p = 6 LED pairs and 6*3 = 18 equations hence
         * the length of F is 18.
         *
         */
        void createF(const gsl_vector *x, gsl_vector *F) const;

        /*
         * The Jacobian matrix is matrix with the same number of rows as F has
         * And the number of columns is the number of found glints.
         * For example, if there are N=4 LEDs:
         *
         *     p = 6
         *     rows = 3*p = 3*6 = 18
         *     columns = N = 4
         *
         */
        void createJacobian(const gsl_vector *gx,
                            gsl_matrix *J) const;


        size_t getNofData() const {return data.size();}

    private:

        /*
         * The copy constructor and assignment operator are prohibited.
         */
        Cornea(const Cornea &other);
        void operator=(const Cornea &other);


        void create(const std::vector<Eigen::Vector3d,
                    Eigen::aligned_allocator<Eigen::Vector3d> > &led_pos,    // LED locations
                    const std::vector<Eigen::Vector3d,
                    Eigen::aligned_allocator<Eigen::Vector3d> > &glint_pos); // corresponding glint locations

        /*
         *	Creates a rotation matrix which describes how a point in an auxiliary
         *	coordinate system, whose x axis is desbibed by vec_along_x_axis and has
         *	a point on its xz-plane vec_on_xz_plane, rotates into the real coordinate
         *	system.
         */
        static void createRotationMatrix(const Eigen::Vector3d &vec_along_x_axis,
                                         const Eigen::Vector3d &vec_on_xz_plane,
                                         Eigen::Matrix3d &R);

        std::vector<DATA_FOR_CORNEA_COMPUTATION> data;

    };

}    // end of "namespace gt"

#endif

