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
    { motionTypes::piston, "piston" },
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


void Foam::waveMakerZPointPatchVectorField::output()
{
    std::ofstream filestream;
    filestream.open("spectra", std::ios::out);
    filestream << "% w ; k ; k2w ; free ; bound ;"
               << " wavemaker(1) ; wavemaker d (2) ; wavemaker b (2) ;"
               << " wavemaker total (1+2)";
    for (long unsigned int j = 0; j < w_.size(); ++j)
    {
        filestream << "\n" << w_[j] << " " << k_[j] << " " << k2w_[j] << " "
                           << abs(Aa_[j]) << " " << abs(AbPF_[j]) << " "
                           << abs(AX1_[j]) << " " << abs(AX2d_[j]) << " "
                           << abs(AX2b_[j]) << " " << abs(AX12_[j]);
    }
    filestream.close();

    filestream.open("time_series", std::ios::out);
    filestream << "% time ; free ; bound ;"
               << " wavemaker(1) ; wavemaker d (2) ; wavemaker b (2) ;"
               << " wavemaker total (1+2)";
    for (long unsigned int j = 0; j < inputTime_.size(); ++j)
    {
        filestream << "\n" << inputTime_[j] << " "
                           << inputElevation_[j] << " "
                           << elevationBound_[j] << " "
                           << XX1_[j] << " " << XX2d_[j] << " "
                           << XX2b_[j] << " " << XX12_[j];
    }
    filestream.close();
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

    Aa_.clear();
    Aa_.resize(numHarmonics_+1, std::complex<double>(0.0, 0.0));

    long unsigned int nBy2 = round(0.5 * n);
    for (long unsigned int j = 1; j < Aa_.size(); ++j)
    {
        if (j < nBy2)
        {
            Aa_[j] = std::complex<double>(fft_outp[j][REAL] / nBy2,
                                          fft_outp[j][IMAG] / nBy2);
        }
        else
        {
            Aa_[j] = std::complex<double>(0.0, 0.0);
        }
    }

    w_.clear();
    w_.resize(numHarmonics_+1, 0.0);

    for (long unsigned int j = 1; j < Aa_.size(); ++j)
    {
        Aa_[j] = std::conj(Aa_[j]);
        w_[j] = 2.0 * M_PI * j / (inputTime_.back() -
                                  inputTime_.front());
    }

    // Calculate wave numbers
    k_.clear();
    k_.resize(numHarmonics_+1, 0.0);
    k2w_.clear();
    k2w_.resize(numHarmonics_+1, 0.0);
    for (long unsigned int j = 1; j < w_.size(); ++j)
    {
        double k1 = 0.0;
        double k2 = w_[j] * w_[j] / grav_;
        int iter = 0;
        while ((fabs(k2 - k1) > 1e-5) && (iter < 1000))
        {
            k1 = k2;
            k2 = w_[j] * w_[j] / (grav_ * tanh(k1 * depth_));
            ++iter;
        }
        k_[j] = k2;

        k1 = 0.0;
        k2 = 4.0 * w_[j] * w_[j] / grav_;
        iter = 0;
        while ((fabs(k2 - k1) > 1e-5) && (iter < 1000))
        {
            k1 = k2;
            k2 = 4.0 * w_[j] * w_[j] / (grav_ * tanh(k1 * depth_));
            ++iter;
        }
        k2w_[j] = k2;
    }

    // Calculate 2nd-order bound waves
    Ab_.clear();
    wb_.clear();
    kb_.clear();
    hb_.clear();

    for (long unsigned int j = 1; j < w_.size(); ++j)
    for (long unsigned int m = 1; m < w_.size(); ++m)
    {
        double ki = k_[j] + k_[m];
        double wi = sqrt(grav_ * ki * tanh(ki * depth_));
        std::complex<double> bnd =
                - V11(wi, w_[j], w_[m], ki, k_[j], k_[m])
                / (wi - w_[j] - w_[m])
                * (M_PI * sqrt(2.0 * grav_ / w_[j]) * Aa_[j])
                * (M_PI * sqrt(2.0 * grav_ / w_[m]) * Aa_[m]);
        std::complex<double> AbZ =
                  (1.0 / M_PI) * sqrt(wi / (2.0 * grav_)) * bnd;
        hb_.push_back(j + m);
        wb_.push_back(w_[j] + w_[m]);
        kb_.push_back(ki);
        Ab_.push_back(AbZ);

        if (j != m)
        {
            ki = - k_[j] + k_[m];
            wi = sqrt(grav_ * ki * tanh(ki * depth_));
            bnd = - V22(wi, w_[j], w_[m], ki, k_[j], k_[m])
                  / (wi + w_[j] - w_[m])
                  * std::conj(M_PI * sqrt(2.0 * grav_ / w_[j]) * Aa_[j])
                  * (M_PI * sqrt(2.0 * grav_ / w_[m]) * Aa_[m]);
            AbZ = (1.0 / M_PI) * sqrt(wi / (2.0 * grav_)) * bnd;
            hb_.push_back(- j + m);
            wb_.push_back(- w_[j] + w_[m]);
            kb_.push_back(ki);
            Ab_.push_back(AbZ);
        }

        ki = - k_[j] - k_[m];
        wi = sqrt(grav_ * ki * tanh(ki * depth_));
        bnd = - V33(wi, w_[j], w_[m], ki, k_[j], k_[m])
              / (wi + w_[j] + w_[m])
              * std::conj(M_PI * sqrt(2.0 * grav_ / w_[j]) * Aa_[j])
              * std::conj(M_PI * sqrt(2.0 * grav_ / w_[m]) * Aa_[m]);
        AbZ = (1.0 / M_PI) * sqrt(wi / (2.0 * grav_)) * bnd;
        hb_.push_back(- j - m);
        wb_.push_back(- w_[j] - w_[m]);
        kb_.push_back(ki);
        Ab_.push_back(AbZ);
    }

    // Convert bound wave spectrum to positive-frequecy
    AbPF_.clear();
    AbPF_.resize(numHarmonics_+1, std::complex<double>(0.0, 0.0));
    for (long unsigned int j = 0; j < hb_.size(); ++j)
    {
        if (abs(hb_[j]) <= numHarmonics_)
        {
            if (hb_[j] >= 0)
            {
                AbPF_[hb_[j]] = AbPF_[hb_[j]] + Ab_[j];
            }
            else
            {
                AbPF_[abs(hb_[j])] = AbPF_[abs(hb_[j])] + std::conj(Ab_[j]);
            }
        }
    }

    elevationBound_.clear();
    elevationBound_.resize(inputTime_.size(), 0.0);
    for (long unsigned int j = 0; j < inputTime_.size(); ++j)
    for (long unsigned int m = 0; m < hb_.size(); ++m)
    {
        elevationBound_[j] = elevationBound_[j] +
      + real(Ab_[m] * exp(- i1 * wb_[m] * inputTime_[j]));
    }
}


void Foam::waveMakerZPointPatchVectorField::wavemakerPiston()
{
    // Calculate first-order wavemaker motion
    AX1_.clear();
    AX1_.resize(numHarmonics_+1, std::complex<double>(0.0, 0.0));

    std::vector<double> Lambda1;
    Lambda1.resize(numHarmonics_+1, 0.0);

    for (long unsigned int j = 1; j < w_.size(); ++j)
    {
        double kappa1 = grav_ * k_[j] / (w_[j] * cosh(k_[j] * depth_));
        Lambda1[j] = (kappa1 / 2.0) * (cosh(k_[j] * depth_)
                   + (k_[j] * depth_) / sinh(k_[j] * depth_));
        std::complex<double> AdX1 = Lambda1[j] * Aa_[j];
        AX1_[j] = i1 * AdX1 / w_[j];
    }

    XX1_.clear();
    XX1_.resize(inputTime_.size(), 0.0);

    for (long unsigned int j = 0; j < inputTime_.size(); ++j)
    for (long unsigned int m = 1; m < w_.size(); ++m)
    {
        XX1_[j] = XX1_[j] + real(AX1_[m] * exp(- i1 * w_[m] * inputTime_[j]));
    }

    // Second-order correction due to finite displacements of the wavemaker
    AX2d_.clear();
    AX2d_.resize(numHarmonics_+1, std::complex<double>(0.0, 0.0));

    for (long unsigned int j = 1; j < w_.size(); ++j)
    {
        int j2 = j * 2;
        if (j2 <= numHarmonics_)
        {
            std::complex<double> AdX2d = - Lambda1[j] * (grav_ / 2.0)
                * k2w_[j] / (k2w_[j] * k2w_[j] - k_[j] * k_[j])
                * sqr(k_[j] / w_[j])
                * (k2w_[j] - k_[j] * cosh(k2w_[j] * depth_)
                                   / sinh(k2w_[j] * depth_)
                * tanh(k_[j] * depth_)) * Aa_[j] * Aa_[j];
            AX2d_[j2] = i1 * AdX2d / w_[j];
        }
    }

    XX2d_.clear();
    XX2d_.resize(inputTime_.size(), 0.0);

    for (long unsigned int j = 0; j < inputTime_.size(); ++j)
    for (long unsigned int m = 1; m < w_.size(); ++m)
    {
        XX2d_[j] = XX2d_[j] + real(AX2d_[m] * exp(- i1 * w_[m] * inputTime_[j]));
    }

    // Second-order correction due to bound waves
    AX2b_.clear();
    AX2b_.resize(numHarmonics_+1, std::complex<double>(0.0, 0.0));

    for (long unsigned int j = 0; j < hb_.size(); ++j)
    {
        double kappa2 = 3 * wb_[j] / (sinh(kb_[j] * depth_)
                                   * (2.0 + cosh(kb_[j] * depth_)));
        double Lambda2 = (kappa2 / 2.0) * (cosh(kb_[j] * depth_)
                       + (kb_[j] * depth_) / sinh(kb_[j] * depth_));

        std::complex<double> AdX2bZ = Lambda2 * Ab_[j];
        std::complex<double> AX2bZ = i1 * AdX2bZ / wb_[j];

        if (abs(hb_[j]) <= numHarmonics_)
        {
            if (hb_[j] >= 0)
            {
                AX2b_[hb_[j]] = AX2b_[hb_[j]] + AX2bZ;
            }
            else
            {
                AX2b_[abs(hb_[j])] = AX2b_[abs(hb_[j])] + std::conj(AX2bZ);
            }
        }
    }

    XX2b_.clear();
    XX2b_.resize(inputTime_.size(), 0.0);

    for (long unsigned int j = 0; j < inputTime_.size(); ++j)
    for (long unsigned int m = 1; m < w_.size(); ++m)
    {
        XX2b_[j] = XX2b_[j] + real(AX2b_[m] * exp(- i1 * w_[m] * inputTime_[j]));
    }

    // Total wavemaker motion
    AX12_.clear();
    AX12_.resize(numHarmonics_+1, std::complex<double>(0.0, 0.0));

    for (long unsigned int j = 1; j < w_.size(); ++j)
    {
        AX12_[j] = AX1_[j] + AX2d_[j] + AX2b_[j];
    }

    XX12_.clear();
    XX12_.resize(inputTime_.size(), 0.0);

    for (long unsigned int j = 0; j < inputTime_.size(); ++j)
    for (long unsigned int m = 1; m < w_.size(); ++m)
    {
        XX12_[j] = XX12_[j] + real(AX12_[m] * exp(- i1 * w_[m] * inputTime_[j]));
    }
}


void Foam::waveMakerZPointPatchVectorField::wavemakerHinged()
{
    // Calculate first-order wavemaker motion
    AX1_.clear();
    AX1_.resize(numHarmonics_+1, std::complex<double>(0.0, 0.0));

    std::vector<double> Lambda1;
    Lambda1.resize(numHarmonics_+1, 0.0);

    for (long unsigned int j = 1; j < w_.size(); ++j)
    {
        double kappa1 = grav_ * k_[j] / (w_[j] * cosh(k_[j] * depth_));
        Lambda1[j] = kappa1 * k_[j] * (depth_ + hinge_) 
                   * (sinh(2.0 * k_[j] * depth_) +2.0 * k_[j] * depth_)
                   / (4.0 * (1.0 - cosh(k_[j] * depth_)
                   + k_[j] * (depth_ + hinge_) * sinh(k_[j] * depth_)));
        std::complex<double> AdX1 = Lambda1[j] * Aa_[j];
        AX1_[j] = i1 * AdX1 / w_[j];
    }

    XX1_.clear();
    XX1_.resize(inputTime_.size(), 0.0);

    for (long unsigned int j = 0; j < inputTime_.size(); ++j)
    for (long unsigned int m = 1; m < w_.size(); ++m)
    {
        XX1_[j] = XX1_[j] + real(AX1_[m] * exp(- i1 * w_[m] * inputTime_[j]));
    }

    // Second-order correction due to finite displacements of the wavemaker
    AX2d_.clear();
    AX2d_.resize(numHarmonics_+1, std::complex<double>(0.0, 0.0));

    for (long unsigned int j = 1; j < w_.size(); ++j)
    {
        int j2 = j * 2;
        if (j2 <= numHarmonics_)
        {
            double kappa_d = Lambda1[j] * grav_ * k_[j]
                           / (2.0 * (depth_ + hinge_) * w_[j] * w_[j]
                           * cosh(k_[j] * depth_));
            double G = k2w_[j] / sqr(k_[j] - k2w_[j])
                     - k2w_[j] / sqr(k_[j] + k2w_[j])
                     + k2w_[j] * cosh((k2w_[j] + k_[j]) * depth_)
                     / sqr(k_[j] + k2w_[j])
                     + k_[j] * (depth_ + hinge_)
                     * sinh((k_[j] + k2w_[j]) * depth_) / (k_[j] + k2w_[j])
                     + k_[j] * (k_[j] - k2w_[j]) * (depth_ + hinge_)
                     * sinh((k_[j] - k2w_[j]) * depth_)
                     / sqr(k_[j] - k2w_[j])
                     - k2w_[j] * cosh((k_[j] - k2w_[j]) * depth_)
                     / sqr(k_[j] - k2w_[j]);
            std::complex<double> AdX2d = - kappa_d * k2w_[j] * k2w_[j]
                     * (depth_ + hinge_) / (
                          2.0 * (1.0 - cosh(k2w_[j] * depth_)
                        + k2w_[j] * (depth_ + hinge_) * sinh(k2w_[j] * depth_))
                     ) * G * Aa_[j] * Aa_[j];
            AX2d_[j2] = i1 * AdX2d / w_[j];
        }
    }

    XX2d_.clear();
    XX2d_.resize(inputTime_.size(), 0.0);

    for (long unsigned int j = 0; j < inputTime_.size(); ++j)
    for (long unsigned int m = 1; m < w_.size(); ++m)
    {
        XX2d_[j] = XX2d_[j] + real(AX2d_[m] * exp(- i1 * w_[m] * inputTime_[j]));
    }

    // Second-order correction due to bound waves
    AX2b_.clear();
    AX2b_.resize(numHarmonics_+1, std::complex<double>(0.0, 0.0));

    for (long unsigned int j = 0; j < hb_.size(); ++j)
    {
        double kappa2 = 3 * wb_[j] / (sinh(kb_[j] * depth_)
                                   * (2.0 + cosh(kb_[j] * depth_)));
        double Lambda2 = kappa2 * kb_[j] * (depth_ + hinge_)
                       * (sinh(2.0 * kb_[j] * depth_) + 2.0 * kb_[j] * depth_)
                       / (4.0 * (1.0 - cosh(kb_[j] * depth_)
                       + kb_[j] * (depth_ + hinge_) * sinh(kb_[j] * depth_)));
        std::complex<double> AdX2bZ = Lambda2 * Ab_[j];
        std::complex<double> AX2bZ = i1 * AdX2bZ / wb_[j];

        if (abs(hb_[j]) <= numHarmonics_)
        {
            if (hb_[j] >= 0)
            {
                AX2b_[hb_[j]] = AX2b_[hb_[j]] + AX2bZ;
            }
            else
            {
                AX2b_[abs(hb_[j])] = AX2b_[abs(hb_[j])] + std::conj(AX2bZ);
            }
        }
    }

    XX2b_.clear();
    XX2b_.resize(inputTime_.size(), 0.0);

    for (long unsigned int j = 0; j < inputTime_.size(); ++j)
    for (long unsigned int m = 1; m < w_.size(); ++m)
    {
        XX2b_[j] = XX2b_[j] + real(AX2b_[m] * exp(- i1 * w_[m] * inputTime_[j]));
    }

    // Total wavemaker motion
    AX12_.clear();
    AX12_.resize(numHarmonics_+1, std::complex<double>(0.0, 0.0));

    for (long unsigned int j = 1; j < w_.size(); ++j)
    {
        AX12_[j] = AX1_[j] + AX2d_[j] + AX2b_[j];
    }

    XX12_.clear();
    XX12_.resize(inputTime_.size(), 0.0);

    for (long unsigned int j = 0; j < inputTime_.size(); ++j)
    for (long unsigned int m = 1; m < w_.size(); ++m)
    {
        XX12_[j] = XX12_[j] + real(AX12_[m] * exp(- i1 * w_[m] * inputTime_[j]));
    }
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::waveMakerZPointPatchVectorField::waveMakerZPointPatchVectorField
(
    const pointPatch& p,
    const DimensionedField<vector, pointMesh>& iF
)
:
    fixedValuePointPatchField<vector>(p, iF),
    grav_(0),
    motionType_(motionTypes::piston),
    secondOrder_(false),
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
    Aa_.clear();
    w_.clear();
    k_.clear();
    k2w_.clear();
    Ab_.clear();
    wb_.clear();
    kb_.clear();
    hb_.clear();
    AbPF_.clear();
    elevationBound_.clear();
    AX1_.clear();
    AX2d_.clear();
    AX2b_.clear();
    AX12_.clear();
    XX1_.clear();
    XX2d_.clear();
    XX2b_.clear();
    XX12_.clear();
}


Foam::waveMakerZPointPatchVectorField::waveMakerZPointPatchVectorField
(
    const pointPatch& p,
    const DimensionedField<vector, pointMesh>& iF,
    const dictionary& dict
)
:
    fixedValuePointPatchField<vector>(p, iF, dict, false),
    grav_(0),
    motionType_(motionTypeNames.lookup("motionType", dict)),
    secondOrder_(dict.get<bool>("secondOrder")),
    n_(dict.get<vector>("n")),
    gHat_(Zero),
    depth_(dict.get<scalar>("depth")),
    rampTime_(dict.get<scalar>("rampTime")),
    hinge_(0),
    numHarmonics_(dict.get<scalar>("numHarmonics")),
    inputFile_(dict.lookup("inputFile"))
{
    Aa_.clear();
    w_.clear();
    k_.clear();
    k2w_.clear();
    Ab_.clear();
    wb_.clear();
    kb_.clear();
    hb_.clear();
    AbPF_.clear();
    elevationBound_.clear();
    AX1_.clear();
    AX2d_.clear();
    AX2b_.clear();
    AX12_.clear();
    XX1_.clear();
    XX2d_.clear();
    XX2b_.clear();
    XX12_.clear();

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
        // Info << "\nReading file : " << filename << endl << endl;
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

    if (motionType_ == motionTypes::hinged)
    {
        wavemakerHinged();
    }
    else if (motionType_ == motionTypes::piston)
    {
        wavemakerPiston();
    }
    else
    {
        FatalErrorInFunction
                << "Unhandled enumeration " << motionTypeNames[motionType_]
                << abort(FatalError);
    }

    output();

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
    grav_(ptf.grav_),
    motionType_(ptf.motionType_),
    secondOrder_(ptf.secondOrder_),
    n_(ptf.n_),
    gHat_(ptf.gHat_),
    depth_(ptf.depth_),
    rampTime_(ptf.rampTime_),
    hinge_(ptf.hinge_),
    numHarmonics_(ptf.numHarmonics_),
    inputFile_(ptf.inputFile_),
    inputTime_(ptf.inputTime_),
    inputElevation_(ptf.inputElevation_),
    Aa_(ptf.Aa_),
    w_(ptf.w_),
    k_(ptf.k_),
    k2w_(ptf.k2w_),
    Ab_(ptf.Ab_),
    wb_(ptf.wb_),
    kb_(ptf.kb_),
    hb_(ptf.hb_),
    AbPF_(ptf.AbPF_),
    elevationBound_(ptf.elevationBound_),
    AX1_(ptf.AX1_),
    AX2d_(ptf.AX2d_),
    AX2b_(ptf.AX2b_),
    AX12_(ptf.AX12_),
    XX1_(ptf.XX1_),
    XX2d_(ptf.XX2d_),
    XX2b_(ptf.XX2b_),
    XX12_(ptf.XX12_)
{}


Foam::waveMakerZPointPatchVectorField::waveMakerZPointPatchVectorField
(
    const waveMakerZPointPatchVectorField& ptf,
    const DimensionedField<vector, pointMesh>& iF
)
:
    fixedValuePointPatchField<vector>(ptf, iF),
    grav_(ptf.grav_),
    motionType_(ptf.motionType_),
    secondOrder_(ptf.secondOrder_),
    n_(ptf.n_),
    gHat_(ptf.gHat_),
    depth_(ptf.depth_),
    rampTime_(ptf.rampTime_),
    hinge_(ptf.hinge_),
    numHarmonics_(ptf.numHarmonics_),
    inputFile_(ptf.inputFile_),
    inputTime_(ptf.inputTime_),
    inputElevation_(ptf.inputElevation_),
    Aa_(ptf.Aa_),
    w_(ptf.w_),
    k_(ptf.k_),
    k2w_(ptf.k2w_),
    Ab_(ptf.Ab_),
    wb_(ptf.wb_),
    kb_(ptf.kb_),
    hb_(ptf.hb_),
    AbPF_(ptf.AbPF_),
    elevationBound_(ptf.elevationBound_),
    AX1_(ptf.AX1_),
    AX2d_(ptf.AX2d_),
    AX2b_(ptf.AX2b_),
    AX12_(ptf.AX12_),
    XX1_(ptf.XX1_),
    XX2d_(ptf.XX2d_),
    XX2b_(ptf.XX2b_),
    XX12_(ptf.XX12_)
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::waveMakerZPointPatchVectorField::updateCoeffs()
{
    if (this->updated())
    {
        return;
    }

    const scalar t = db().time().value();

    scalar motionX = 0.0;

    for (long unsigned int j = 1; j < w_.size(); ++j)
    {
        if (secondOrder_)
        {
            motionX = motionX + real(AX12_[j] * exp(- i1 * w_[j] * t));
        }
        else
        {
            motionX = motionX + real(AX1_[j] * exp(- i1 * w_[j] * t));
        }
    }

    switch (motionType_)
    {
        case motionTypes::hinged:
        {
            const pointField& points = patch().localPoints();
            const scalarField shapeFunc
                  (1 - (points & gHat_) / (hinge_ + depth_));

            Field<vector>::operator=
            (
                n_ * timeCoeff(t) * motionX * shapeFunc
            );

            break;
        }
        case motionTypes::piston:
        {
            Field<vector>::operator=
            (
                n_ * timeCoeff(t) * motionX
            );

            break;
        }
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
    os.writeEntry("secondOrder", secondOrder_);
    os.writeEntry("n", n_);
    os.writeEntry("depth", depth_);
    os.writeEntry("rampTime", rampTime_);
    os.writeEntry("hingeLocation", hinge_);
    os.writeEntry("numHarmonics", numHarmonics_);
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
