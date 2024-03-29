/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2018 OpenCFD Ltd.
     \\/     M anipulation  | Copyright (C) 2018 IH-Cantabria
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Class
    Foam::waveMakerZPointPatchVectorField

Description
    Point motion boundary condition to generate waves based on either piston
    or hinged motions.

    The second-order correction to the wavemaker motion is based on
    the Zakharov theory.

    If you use this library please cite our papers:
    [1] A. Khait, L. Shemer.
        Nonlinear wave generation by a wavemaker in deep to intermediate
        water depth. Ocean Engineering 182 (2019) 222–234
    [2] Anatoliy Khait, Lev Shemer.
        Nonlinear generation of narrow-banded wave trains.
        OMAE2019-95364

    Dr. Anatoliy Khait (haitanatoliy@gmail.com)
    Prof. Lev Shemer (shemerl@tauex.tau.ac.il ; shemer@eng.tau.ac.il)
    Dr. Zhihua Ma (z.ma@mmu.ac.uk)

Usage
    Example patch specification
    leftwall
    {
        type            waveMakerZ;
        value           uniform (0 0 0);

        motionType      hinged;
        secondOrder     true;
        n               (1 0 0);
        depth           0.75;
        rampTime        3.0;
        hingeLocation   0.75;
        numHarmonics    32;
        inputFile       "signal";
    }

    where
    \table
        Property      | Description                   | Required | Default
        --------------|-------------------------------|----------|---------
        motionType    | See motion types below        | yes      |
        secondOrder   | Second-order correction       | yes      |
        n             | Direction of motion           | yes      |
        depth         | Wave tank depth               | yes      |
        rampTime      | Time to reach maximum motion  | yes      |
        hingeLocation | Location of the hinge         | yes      |
        numHarmonics  | Number of frequency harmonics | yes      |
                      |              to be considered |          |
        inputFile     | File with the 1st-order       | yes      |
                      |             surface elevation |          |
    \endtable

    Available motion types include:
    - piston
    - hinged

SourceFiles
    waveMakerZPointPatchVectorField.C

\*---------------------------------------------------------------------------*/

#ifndef waveMakerZPointPatchVectorField_H
#define waveMakerZPointPatchVectorField_H

#define REAL 0
#define IMAG 1

#include "fixedValuePointPatchField.H"
#include "Enum.H"
#include <fftw3.h>
#include <complex>
#include <vector>
#include <fstream>

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
               Class waveMakerZPointPatchVectorField Declaration
\*---------------------------------------------------------------------------*/

class waveMakerZPointPatchVectorField
:
    public fixedValuePointPatchField<vector>
{
    public:

        enum motionTypes
        {
            piston,
            hinged
        };

        //- Names for motion types
        static const Enum<motionTypes> motionTypeNames;


    // Private data
    private:

        const std::complex<double> i1 = std::complex<double>(0.0, 1.0);

        //- Absolute value of gravity
        scalar grav_;

        //- Motion type
        motionTypes motionType_;

        //- Type of wave generation
        bool secondOrder_;

        //- Patch normal direction
        //  Note: cannot use patch normal of the initial patch unless it is
        //  in its neutral position (flap mode)
        vector n_;

        //- Vertical direction
        vector gHat_;

        //- Inital water depth
        scalar depth_;

        //- Ramp time
        scalar rampTime_;

        //- Location of the hinge
        scalar hinge_;

        //- Number of frequencies in spectrum
        label numHarmonics_;

        //- File of the input signal
        string inputFile_;

        //- Time in input signal
        std::vector<double> inputTime_;

        //- Surface elevation in input signal
        std::vector<double> inputElevation_;

    // -------------------------------------------------------------------------

        //- Complex spectrum of free waves
        std::vector<std::complex<double>> Aa_;

        //- Angular frequency
        std::vector<double> w_;

        //- Wave number
        std::vector<double> k_;

        //- k2w = k(2*w)
        std::vector<double> k2w_;

        //- Complex spectrum of bound waves
        std::vector<std::complex<double>> Ab_;

        //- Angular frequency of bound waves
        std::vector<double> wb_;

        //- Wave number of bound waves
        std::vector<double> kb_;

        //- Number of bound wave frequency harmonic
        std::vector<int> hb_;

        //- Complex spectrum of bound waves
        //  in positive-frequecy representation
        std::vector<std::complex<double>> AbPF_;

        //- Surface elevation of bound waves
        std::vector<double> elevationBound_;

        //- Spectrum of first-order wavemaker displacements
        std::vector<std::complex<double>> AX1_;

        //- Second-order spectrum due to finite displacements
        std::vector<std::complex<double>> AX2d_;

        //- Second-order spectrum due to bound waves
        std::vector<std::complex<double>> AX2b_;

        //- Spectrum of total wavemaker motion
        std::vector<std::complex<double>> AX12_;

        //- First-order wavemaker displacements
        std::vector<double> XX1_;

        //- Second-order motion due to finite displacements
        std::vector<double> XX2d_;

        //- Second-order motion due to bound waves
        std::vector<double> XX2b_;

        //- Total wavemaker motion
        std::vector<double> XX12_;

    // Protected Member Functions

        //- Return the gravitational acceleration
        const vector& g();

        //- Return the time scaling coefficient
        virtual scalar timeCoeff(const scalar t) const;

        //- Kernels of Zakharov theory
        Foam::scalar V012
        (
            const scalar w0,
            const scalar w1,
            const scalar w2,
            const scalar k0,
            const scalar k1
        );
        Foam::scalar V11
        (
            const scalar w0,
            const scalar w1,
            const scalar w2,
            const scalar k0,
            const scalar k1,
            const scalar k2
        );
        Foam::scalar V22
        (
            const scalar w0,
            const scalar w1,
            const scalar w2,
            const scalar k0,
            const scalar k1,
            const scalar k2
        );
        Foam::scalar V33
        (
            const scalar w0,
            const scalar w1,
            const scalar w2,
            const scalar k0,
            const scalar k1,
            const scalar k2
        );

        //- Output results of calculations into text files
        void output();

        //- Read surface elevation from file
        //  Construct free wave spectrum
        //  Calculate second-order bound wave spectrum
        void spectrum();

        //- Calculate first- and second-order wavemaker motion
        void wavemakerHinged();
        void wavemakerPiston();


    public:

        //- Runtime type information
        TypeName("waveMakerZ");


        // Constructors

        //- Construct from patch and internal field
        waveMakerZPointPatchVectorField
        (
            const pointPatch&,
            const DimensionedField<vector, pointMesh>&
        );

        //- Construct from patch, internal field and dictionary
        waveMakerZPointPatchVectorField
        (
            const pointPatch&,
            const DimensionedField<vector, pointMesh>&,
            const dictionary&
        );

        //- Construct by mapping given patchField<vector> onto a new patch
        waveMakerZPointPatchVectorField
        (
            const waveMakerZPointPatchVectorField&,
            const pointPatch&,
            const DimensionedField<vector, pointMesh>&,
            const pointPatchFieldMapper&
        );

        //- Construct and return a clone
        virtual autoPtr<pointPatchField<vector>> clone() const
        {
            return autoPtr<pointPatchField<vector>>
            (
                new waveMakerZPointPatchVectorField
                (
                    *this
                )
            );
        }

        //- Construct as copy setting internal field reference
        waveMakerZPointPatchVectorField
        (
            const waveMakerZPointPatchVectorField&,
            const DimensionedField<vector, pointMesh>&
        );

        //- Construct and return a clone setting internal field reference
        virtual autoPtr<pointPatchField<vector>> clone
        (
            const DimensionedField<vector, pointMesh>& iF
        ) const
        {
            return autoPtr<pointPatchField<vector>>
            (
                new waveMakerZPointPatchVectorField
                (
                    *this,
                    iF
                )
            );
        }


    // Member functions

        // Evaluation functions

        //- Update the coefficients associated with the patch field
        virtual void updateCoeffs();

        //- Write
        virtual void write(Ostream&) const;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
