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
    or flap motions.

    Based on the reference
    \verbatim
        Hughes, S.A. (1993).
        Physical Models And Laboratory Techniques In Coastal Engineering.
        Advanced Series On Ocean Engineering, volume 7
    \endverbatim

Usage
    Example patch specification
    \verbatim
    leftwall
    {
        type            waveMaker; 
        motionType      flap;
        n               (1 0 0);
        initialDepth    0.25;
        wavePeriod      2.0;
        waveHeight      0.06;
        wavePhase       0;
        rampTime        2.0;
    }
    \endverbatim

    where
    \table
        Property     | Description                  | Required | Default value
        motionType   | See motion types below       | yes      |
        x0           | Reference point on patch     | yes      |
        n            | Direction of motion          | yes      |
        initialDepth | Initial depth                | yes      |
        wavePeriod   | wave period                  | yes      |
        waveHeight   | Wave height                  | yes      |
        wavePhase    | wave phase                   | yes      |
        rampTime     | Time to reach maximum motion | yes      |
        secondOrder  | Second order calculation     | no       | no
    \endtable

    Available motion types include:
    - piston
    - flap

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
            // piston,
            hinged
        };

        //- Names for motion types
        static const Enum<motionTypes> motionTypeNames;


    // Private data
    private:

        const std::complex<double> i1 = std::complex<double>(0.0, 1.0);

        //- Motion type
        motionTypes motionType_;

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
        std::vector<std::complex<double>> Aa;

        //- Angular frequency
        std::vector<double> w;

        //- Wave number
        std::vector<double> k;

        //- Complex spectrum of bound waves
        std::vector<std::complex<double>> Ab;

        //- Angular frequency of bound waves
        std::vector<double> wb;

        //- Wave number of bound waves
        std::vector<double> kb;

        //- Number of bound wave frequency harmonic
        std::vector<int> hb;

        //- Complex spectrum of bound waves
        //  in positive-frequecy representation
        std::vector<std::complex<double>> AbPF;


        //- Absolute value of gravity
        scalar grav_;


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


        //- Read surface elevation from file
        void spectrum();


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
