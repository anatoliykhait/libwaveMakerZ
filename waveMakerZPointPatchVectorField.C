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

\*---------------------------------------------------------------------------*/

#include "waveMakerZPointPatchVectorField.H"
#include "mathematicalConstants.H"
#include "pointPatchFields.H"
#include "addToRunTimeSelectionTable.H"
#include "Time.H"
#include "gravityMeshObject.H"

// * * * * * * * * * * * * * Static Member Data  * * * * * * * * * * * * * * //

const Foam::Enum<Foam::waveMakerZPointPatchVectorField::motionTypes>
Foam::waveMakerZPointPatchVectorField::motionTypeNames
({
    // { motionTypes::piston, "piston" },
    { motionTypes::hinged, "hinged" }
});


// * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * * //

const Foam::vector& Foam::waveMakerZPointPatchVectorField::g()
{
    const meshObjects::gravity& gf = meshObjects::gravity::New(db().time());

    if (mag(gf.value()) < SMALL)
    {
        FatalErrorInFunction
            << "Gravity vector is not set.  Please update "
            << gf.uniformDimensionedVectorField::path()
            << exit(FatalError);
    }

    return gf.value();
}


Foam::scalar Foam::waveMakerZPointPatchVectorField::timeCoeff
(
    const scalar t
) const
{
    return max(0, min(t/rampTime_, 1));
}


Foam::scalar Foam::waveMakerZPointPatchVectorField::V012
(
    const scalar w0,
    const scalar w1,
    const scalar w2,
    const scalar k0,
    const scalar k1
)
{
    scalar V = 1.0 / (8.0 * M_PI)
             * sqrt(grav_ * w2 / (2.0 * w0 * w1))
             * (k0 * k1 + sqr(w0 * w1 / grav_));
    return V;
}


Foam::scalar Foam::waveMakerZPointPatchVectorField::V11
(
    const scalar w0,
    const scalar w1,
    const scalar w2,
    const scalar k0,
    const scalar k1,
    const scalar k2
)
{
    scalar V = -2.0 * V012(w0, w1, w2, -k0, k1)
                    + V012(w1, w2, w0, k1, k2);
    return V;
}


Foam::scalar Foam::waveMakerZPointPatchVectorField::V22
(
    const scalar w0,
    const scalar w1,
    const scalar w2,
    const scalar k0,
    const scalar k1,
    const scalar k2
)
{
    scalar V = 2.0 * (V012(w0, w1, w2, k0, k1)
                    - V012(w0, w2, w1, -k0, k2)
                    - V012(w1, w2, w0, -k1, k2));
    return V;
}


Foam::scalar Foam::waveMakerZPointPatchVectorField::V33
(
    const scalar w0,
    const scalar w1,
    const scalar w2,
    const scalar k0,
    const scalar k1,
    const scalar k2
)
{
    scalar V = 2.0 * V012(w0, w1, w2, k0, k1)
                   + V012(w1, w2, w0, k1, k2);
    return V;
}


void Foam::waveMakerZPointPatchVectorField::spectrum()
{
    // Arrays for fft
    int n = inputTime_.size();
    fftw_complex fft_inp[n];
    fftw_complex fft_outp[n];

    for (int j = 0; j < n; ++j)
    {
        fft_inp[j][REAL] = inputElevation_[j];
        fft_inp[j][IMAG] = 0.0;
    }

    fftw_plan plan = fftw_plan_dft_1d
    (
        n, fft_inp, fft_outp, FFTW_FORWARD, FFTW_ESTIMATE
    );
    fftw_execute(plan);
    fftw_destroy_plan(plan);
    fftw_cleanup();

    Aa.clear();
    Aa.resize(numHarmonics_+1, std::complex<double>(0.0, 0.0));

    long unsigned int nBy2 = round(0.5 * n);
    for (long unsigned int j = 1; j < Aa.size(); ++j)
    {
        if (j < nBy2)
        {
            Aa[j] = std::complex<double>(fft_outp[j][REAL] / nBy2,
                                         fft_outp[j][IMAG] / nBy2);
        }
        else
        {
            Aa[j] = std::complex<double>(0.0, 0.0);
        }
    }

    w.clear();
    w.resize(numHarmonics_+1, 0.0);

    for (long unsigned int j = 1; j < Aa.size(); ++j)
    {
        Aa[j] = std::conj(Aa[j]);
        w[j] = 2.0 * M_PI * j / (inputTime_.back() -
                                 inputTime_.front());
    }

    // Calculate wave numbers
    k.clear();
    k.resize(numHarmonics_+1, 0.0);
    for (long unsigned int j = 1; j < w.size(); ++j)
    {
        double k1 = 0.0;
        double k2 = w[j] * w[j] / grav_;
        int iter = 0;
        while ((fabs(k2 - k1) > 1e-5) && (iter < 1000))
        {
            k1 = k2;
            k2 = w[j] * w[j] / (grav_ * tanh(k1 * depth_));
            ++iter;
        }
        k[j] = k2;
    }

    // Calculate 2nd-order bound waves
    Ab.clear();
    wb.clear();
    kb.clear();
    hb.clear();

    for (long unsigned int j = 1; j < w.size(); ++j)
    for (long unsigned int m = 1; m < w.size(); ++m)
    {
        double ki = k[j] + k[m];
        double wi = sqrt(grav_ * ki * tanh(ki * depth_));
        std::complex<double> bnd =
                - V11(wi, w[j], w[m], ki, k[j], k[m])
                / (wi - w[j] - w[m])
                * (M_PI * sqrt(2.0 * grav_ / w[j]) * Aa[j])
                * (M_PI * sqrt(2.0 * grav_ / w[m]) * Aa[m]);
        std::complex<double> AbZ =
                  (1.0 / M_PI) * sqrt(wi / (2.0 * grav_)) * bnd;
        hb.push_back(j + m);
        wb.push_back(w[j] + w[m]);
        kb.push_back(ki);
        Ab.push_back(AbZ);

        if (j != m)
        {
            ki = - k[j] + k[m];
            wi = sqrt(grav_ * ki * tanh(ki * depth_));
            bnd = - V22(wi, w[j], w[m], ki, k[j], k[m])
                  / (wi + w[j] - w[m])
                  * std::conj(M_PI * sqrt(2.0 * grav_ / w[j]) * Aa[j])
                  * (M_PI * sqrt(2.0 * grav_ / w[m]) * Aa[m]);
            AbZ = (1.0 / M_PI) * sqrt(wi / (2.0 * grav_)) * bnd;
            hb.push_back(- j + m);
            wb.push_back(- w[j] + w[m]);
            kb.push_back(ki);
            Ab.push_back(AbZ);
        }

        ki = - k[j] - k[m];
        wi = sqrt(grav_ * ki * tanh(ki * depth_));
        bnd = - V33(wi, w[j], w[m], ki, k[j], k[m])
              / (wi + w[j] + w[m])
              * std::conj(M_PI * sqrt(2.0 * grav_ / w[j]) * Aa[j])
              * std::conj(M_PI * sqrt(2.0 * grav_ / w[m]) * Aa[m]);
        AbZ = (1.0 / M_PI) * sqrt(wi / (2.0 * grav_)) * bnd;
        hb.push_back(- j - m);
        wb.push_back(- w[j] - w[m]);
        kb.push_back(ki);
        Ab.push_back(AbZ);
    }

    // Convert bound wave spectrum to positive-frequecy
    AbPF.clear();
    AbPF.resize(numHarmonics_+1, std::complex<double>(0.0, 0.0));
    for (long unsigned int j = 0; j < hb.size(); ++j)
    {
        if (abs(hb[j]) <= numHarmonics_)
        {
            if (hb[j] >= 0)
            {
                AbPF[hb[j]] = AbPF[hb[j]] + Ab[j];
            }
            else
            {
                AbPF[abs(hb[j])] = AbPF[abs(hb[j])] + std::conj(Ab[j]);
            }
        }
    }

    std::ofstream filestream;
    filestream.open("elevation_spectrum", std::ios::out);
    filestream << "% w ; k ; free ; bound";
    for (long unsigned int j = 0; j < w.size(); ++j)
    {
        filestream << "\n" << w[j] << " " << k[j] << " "
                           << abs(Aa[j]) << " " << abs(AbPF[j]);
    }
    filestream.close();
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::waveMakerZPointPatchVectorField::waveMakerZPointPatchVectorField
(
    const pointPatch& p,
    const DimensionedField<vector, pointMesh>& iF
)
:
    fixedValuePointPatchField<vector>(p, iF),
    motionType_(motionTypes::hinged), // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    n_(Zero),
    gHat_(Zero),
    depth_(0),
    rampTime_(0),
    hinge_(0),
    numHarmonics_(0),
    inputFile_("")
{
    inputTime_.clear();
    inputElevation_.clear();
}


Foam::waveMakerZPointPatchVectorField::waveMakerZPointPatchVectorField
(
    const pointPatch& p,
    const DimensionedField<vector, pointMesh>& iF,
    const dictionary& dict
)
:
    fixedValuePointPatchField<vector>(p, iF, dict, false),
    motionType_(motionTypeNames.lookup("motionType", dict)),
    n_(dict.get<vector>("n")),
    gHat_(Zero),
    depth_(dict.get<scalar>("depth")),
    rampTime_(dict.get<scalar>("rampTime")),
    hinge_(0),
    numHarmonics_(dict.get<scalar>("numHarmonics")),
    inputFile_(dict.lookup("inputFile"))
{
    // Create the co-ordinate system
    if (mag(n_) < SMALL)
    {
        FatalIOErrorInFunction(dict)
            << "Patch normal direction vector is not set.  'n' = " << n_
            << exit(FatalIOError);
    }
    n_ /= mag(n_);

    gHat_ = g() / mag(g());

    grav_ = mag(g());

    if (motionType_ == motionTypes::hinged)
    {
        hinge_ = dict.get<scalar>("hingeLocation");
    }

    { // Read time and surface elevation from input file
        inputTime_.clear();
        inputElevation_.clear();
        char filename[inputFile_.length()];
        sprintf(filename, "%s", inputFile_.c_str());
        Info << "\nReading file : " << filename << endl << endl;
        std::ifstream filestream(filename);
        filestream.seekg(0, std::ios::beg);
        string fileline;
        while (std::getline(filestream, fileline))
        {
            if (fileline[0] != '#')
            {
                unsigned int jline = fileline.find(' ');
                string value1 = "";
                string value2 = "";
                value1.assign(fileline, 0, jline);
                value2.assign(fileline, jline, fileline.length());
                inputTime_.push_back(stod(value1));
                inputElevation_.push_back(stod(value2));
            }
        }
        filestream.close();
    }

    spectrum();

    if (!dict.found("value"))
    {
        updateCoeffs();
    }
}


Foam::waveMakerZPointPatchVectorField::waveMakerZPointPatchVectorField
(
    const waveMakerZPointPatchVectorField& ptf,
    const pointPatch& p,
    const DimensionedField<vector, pointMesh>& iF,
    const pointPatchFieldMapper& mapper
)
:
    fixedValuePointPatchField<vector>(ptf, p, iF, mapper),
    motionType_(ptf.motionType_),
    n_(ptf.n_),
    gHat_(ptf.gHat_),
    depth_(ptf.depth_),
    rampTime_(ptf.rampTime_),
    hinge_(ptf.hinge_),
    numHarmonics_(ptf.numHarmonics_),
    inputFile_(ptf.inputFile_),
    inputTime_(ptf.inputTime_),
    inputElevation_(ptf.inputElevation_)
{}


Foam::waveMakerZPointPatchVectorField::waveMakerZPointPatchVectorField
(
    const waveMakerZPointPatchVectorField& ptf,
    const DimensionedField<vector, pointMesh>& iF
)
:
    fixedValuePointPatchField<vector>(ptf, iF),
    motionType_(ptf.motionType_),
    n_(ptf.n_),
    gHat_(ptf.gHat_),
    depth_(ptf.depth_),
    rampTime_(ptf.rampTime_),
    hinge_(ptf.hinge_),
    numHarmonics_(ptf.numHarmonics_),
    inputFile_(ptf.inputFile_),
    inputTime_(ptf.inputTime_),
    inputElevation_(ptf.inputElevation_)
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::waveMakerZPointPatchVectorField::updateCoeffs()
{
    if (this->updated())
    {
        return;
    }

    const scalar t = db().time().value();

    switch (motionType_)
    {
        case motionTypes::hinged:
        {
            scalar motionX = 0.05*sin(5.0*t);

            const pointField& points = patch().localPoints();
            // const scalarField dz(-(points & gHat_) + hinge_ - initialDepth_);
            const scalarField shapeFunc
                  (1 - (points & gHat_) / (hinge_ + depth_));

            Field<vector>::operator=
            (
                n_*timeCoeff(t)*motionX*shapeFunc
            );

            break;
        }
        // case motionTypes::piston:
        // {
        //     const scalar m1 = 2*(cosh(2*kh) - 1)/(sinh(2*kh) + 2*kh);

        //     scalar motionX = 0.5*waveHeight_/m1*sin(sigma*t);

        //     if (secondOrder_)
        //     {
        //         motionX += 
        //             sqr(waveHeight_)
        //            /(32*initialDepth_)*(3*cosh(kh)
        //            /pow3(sinh(kh)) - 2/m1);
        //     }

        //     Field<vector>::operator=(n_*timeCoeff(t)*motionX);

        //     break;
        // }
        default:
        {
            FatalErrorInFunction
                << "Unhandled enumeration " << motionTypeNames[motionType_]
                << abort(FatalError);
        }
    }
        

    fixedValuePointPatchField<vector>::updateCoeffs();
}


void Foam::waveMakerZPointPatchVectorField::write(Ostream& os) const
{
    pointPatchField<vector>::write(os);
    os.writeEntry("motionType", motionTypeNames[motionType_]);
    os.writeEntry("n", n_);
    os.writeEntry("depth", depth_);
    os.writeEntry("rampTime", rampTime_);
    os.writeEntry("hingeLocation", hinge_);
    os.writeEntry("inputFile", inputFile_);
    writeEntry("value", os);
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
    makePointPatchTypeField
    (
        pointPatchVectorField,
        waveMakerZPointPatchVectorField
    );
}

// ************************************************************************* //
