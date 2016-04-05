#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <curand_kernel.h>
//#include <curand_normal.h>
#include <cuda_device_runtime_api.h>

__global__ void initialConditions(float* vars, int num_param, int num_cells, int cells_per_thread, int* passed) {
	float V = -83.5092;
	float m = 0.0025;
	float h = 0.6945;
	float j = 0.6924;
	float d = 4.2418e-005;
	float f = 0.9697;
	float f2 = 0.9784;
	float fCass = 0.9999;
	float r = 3.2195e-008;
	float s = 1.0000;
	float xs = 0.0038;
	float xr1 = 3.1298e-004;
	float xr2 = 0.4534;
	float Rbar_ryr = 0.9816;
	float Cai = 0.04;
	float Cass = 0.2381;
	float CaSR = 3.6426e+003;
	float Nai = 3.8067e+003;
	float Ki = 1.2369e+005;

	// Within each test, the variables are divided as follows 
	// V(cell1), V(cell2), V(cell3) ... V(cellLast), m(cell1), m(cell2), ... m(cellLast) ... for all 19 parameters
	int idx = threadIdx.x*cells_per_thread;
	int simulations = blockIdx.x;
	int limit = idx + cells_per_thread;
	int idxx = blockIdx.x * num_cells + threadIdx.x;

	for (; idx<limit; idx++) {
		vars[(simulations*num_param*num_cells) + idx + (0 * num_cells)] = V;
		vars[(simulations*num_param*num_cells) + idx + (1 * num_cells)] = m;
		vars[(simulations*num_param*num_cells) + idx + (2 * num_cells)] = h;
		vars[(simulations*num_param*num_cells) + idx + (3 * num_cells)] = j;
		vars[(simulations*num_param*num_cells) + idx + (4 * num_cells)] = d;
		vars[(simulations*num_param*num_cells) + idx + (5 * num_cells)] = f;
		vars[(simulations*num_param*num_cells) + idx + (6 * num_cells)] = f2;
		vars[(simulations*num_param*num_cells) + idx + (7 * num_cells)] = fCass;
		vars[(simulations*num_param*num_cells) + idx + (8 * num_cells)] = r;
		vars[(simulations*num_param*num_cells) + idx + (9 * num_cells)] = s;
		vars[(simulations*num_param*num_cells) + idx + (10 * num_cells)] = xs;
		vars[(simulations*num_param*num_cells) + idx + (11 * num_cells)] = xr1;
		vars[(simulations*num_param*num_cells) + idx + (12 * num_cells)] = xr2;
		vars[(simulations*num_param*num_cells) + idx + (13 * num_cells)] = Rbar_ryr;
		vars[(simulations*num_param*num_cells) + idx + (14 * num_cells)] = Cai;
		vars[(simulations*num_param*num_cells) + idx + (15 * num_cells)] = Cass;
		vars[(simulations*num_param*num_cells) + idx + (16 * num_cells)] = CaSR;
		vars[(simulations*num_param*num_cells) + idx + (17 * num_cells)] = Nai;
		vars[(simulations*num_param*num_cells) + idx + (18 * num_cells)] = Ki;
	}

	passed[idxx] = 0;

}

__global__ void computeState(float* x, float* ion_current, int total_cells, float step, float* randNums, int variations, float* x_temp, int num_changing_vars, int cells_per_thread) {
	int idx = cells_per_thread*threadIdx.x;
	int cell_num, cell_current_idx;
	int limit = idx + cells_per_thread;
	for (; idx<limit; idx++) {
		cell_num = (blockIdx.x*total_cells * 19) + idx;
		cell_current_idx = (blockIdx.x * total_cells) + idx;

		//Index Variables to make life easier
		//Array  is categorized by blocks of size=total_cells, each block contains the values of one parameter across the cells
		int V_i = 0 * total_cells;
		int m_i = 1 * total_cells;
		int h_i = 2 * total_cells;
		int j_i = 3 * total_cells;
		int d_i = 4 * total_cells;
		int f_i = 5 * total_cells;
		int f2_i = 6 * total_cells;
		int fCass_i = 7 * total_cells;
		int r_i = 8 * total_cells;
		int s_i = 9 * total_cells;
		int xs_i = 10 * total_cells;
		int xr1_i = 11 * total_cells;
		int xr2_i = 12 * total_cells;
		int Rbar_ryr_i = 13 * total_cells;
		int Cai_i = 14 * total_cells;
		int Cass_i = 15 * total_cells;
		int CaSR_i = 16 * total_cells;
		int Nai_i = 17 * total_cells;
		int Ki_i = 18 * total_cells;

		float ENa, EK, ECa, EKs, INa, ICa, Ito, IKs, IKr;
		float aK1, bK1, xK1, IK1;
		float minf, am, bm, taum;
		float hinf, jinf, ad, bd, gd, taud, finf, af, bf, gf, tauf;
		float f2inf, af2, bf2, gf2, tauf2, fCassinf, taufCass;
		float rinf, taur, sinf, taus, xsinf, axs, bxs, tauxs;
		float axr1, bxr1, tauxr1, xr1inf, xr2inf, axr2, bxr2, tauxr2;
		float Ileak, Iup, kcasr, k1_ryr, k2_ryr, O_ryr;
		float Irel, Ixfer, Bi, Bss, BSR;
		float ah, bh, aj, bj;
		float tauh, tauj, dinf, INCX, fNaK, INaK, IpCa, IpK, INab, ICab, Iion;
		char celltype;

		float F = 96.4853415;			// Faraday's constant, coulombs/mmol 
		float R = 8.314472; 				//gas constant, J/(K mol)
		float T = 310.0;				// fabsolute temperature, K 
		float RTF = R*T / F;
		float Acap = 5.6297*3.280e-5;		// cm2
		float Vmyo = 16.404;				// pL
		float VSR = 1.094;				// pL
		float Vss = 0.05468;				// pL
		float Ko = 5400;					// uM
		float Nao = 140000;				// uM
		float Cao = 2000;					// uM

		float PCa_;
		float GNa_;
		float GK1_;
		float GKr_;
		float GpK_;
		float GpCa_;
		float GNab_;
		float GCab_;
		float Gto_;
		float GKs_;
		float INaK_;
		float kNaCa;
		float Vleak;
		float Iup_;
		float Vrel;

		float pKNa = 0.03;				// relative permeability, Na to K
		float KmNa = 87500;				// uM
		float KmCa = 1380;				// uM
		float ksat = 0.1;					// unitless
		float alpha_ncx = 2.5;			// unitless
		float eta = 0.35;					// unitless, actually gamma in paper
		float KmNai = 40000;				// Na-K pump // uM
		float KmKo = 1000;				// Sarcolemmal Ca pump // uM
		float KpCa = 0.5;					// SERCA // uM
		float Kmup = 0.25;				// uM
		float Vxfer = 0.0038;				// ms^-1
		float k1_ryr_prime = 0.15e-6;		// uM-2 ms-1
		float k2_ryr_prime = 0.045e-3;	// uM-1 ms-1
		float k3_ryr = 0.06;				// ms-1
		float k4_ryr = 0.005;				// ms-1 as per KWTT source code
		float maxsr = 2.5;				// dimensionless 
		float minsr = 1;					// dimensionless
		float EC_ryr = 1500;				// uM
		float Bufc = 200;					// uM
		float Kbufc = 1;					// uM
		float Bufss = 400;				// uM
		float Kbufss = 0.25;				// uM
		float BufSR = 10000;				// uM
		float KbufSR = 300;				// uM
		celltype = 'p'; //epi
		//celltype = 'n' ; //endo
		//celltype = 'm' ; //mid

		if (num_changing_vars == 0) {
			GNa_ = 14.838; 				// nS/pF
			GK1_ = 5.405; 				// nS/pF
			GKr_ = 0.153; 				// nS/pF
			GpK_ = 0.0146; 				// nS/pF
			GpCa_ = 0.1238; 			// nS/pF
			GNab_ = 2.9e-4; 				// nS/pF
			GCab_ = 5.92e-4; 			// nS/pF
			if (celltype == 'n') { //endo
				Gto_ = 0.073;
				GKs_ = 0.392;
			}
			else if (celltype == 'p') { //epi
				Gto_ = 0.294;
				GKs_ = 0.392;
			}
			else if (celltype == 'm') { //mid
				Gto_ = 0.294; 				// nS/pF
				GKs_ = 0.098; 				// nS/pF
			}
			PCa_ = 3.980e-5;
			INaK_ = 2.724;				// pA/pF
			kNaCa = 1000;				// pA/pF
			Vleak = 3.6e-4;			// ms^-1
			Iup_ = 6.375;				// uM/ms
			Vrel = 0.102;				// ms^-1 as per KWTT source code
		}
		else {

			GNa_ = 14.838*randNums[(blockIdx.x * total_cells * num_changing_vars) + (threadIdx.x * num_changing_vars) + 0];
			GK1_ = 5.405*randNums[(blockIdx.x * total_cells * num_changing_vars) + (threadIdx.x * num_changing_vars) + 1];
			GKr_ = 0.153*randNums[(blockIdx.x * total_cells * num_changing_vars) + (threadIdx.x * num_changing_vars) + 2];
			GpK_ = 0.0146*randNums[(blockIdx.x * total_cells * num_changing_vars) + (threadIdx.x * num_changing_vars) + 3];
			GpCa_ = 0.1238*randNums[(blockIdx.x * total_cells * num_changing_vars) + (threadIdx.x * num_changing_vars) + 4];
			GNab_ = 2.9e-4*randNums[(blockIdx.x * total_cells * num_changing_vars) + (threadIdx.x * num_changing_vars) + 5];
			GCab_ = 5.92e-4*randNums[(blockIdx.x * total_cells * num_changing_vars) + (threadIdx.x * num_changing_vars) + 6];
			if (celltype == 'n') { //endo
				Gto_ = 0.073*randNums[(blockIdx.x * total_cells * num_changing_vars) + (threadIdx.x * num_changing_vars) + 7];
				GKs_ = 0.392*randNums[(blockIdx.x * total_cells * num_changing_vars) + (threadIdx.x * num_changing_vars) + 8];
			}
			else if (celltype == 'p') { //epi
				Gto_ = 0.294*randNums[(blockIdx.x * total_cells * num_changing_vars) + (threadIdx.x * num_changing_vars) + 7];
				GKs_ = 0.392*randNums[(blockIdx.x * total_cells * num_changing_vars) + (threadIdx.x * num_changing_vars) + 8];
			}
			else if (celltype == 'm') { //mid
				Gto_ = 0.294*randNums[(blockIdx.x * total_cells * num_changing_vars) + (threadIdx.x * num_changing_vars) + 7];
				GKs_ = 0.098*randNums[(blockIdx.x * total_cells * num_changing_vars) + (threadIdx.x * num_changing_vars) + 8];
			}
			PCa_ = 3.980e-5*randNums[(blockIdx.x * total_cells * num_changing_vars) + (threadIdx.x * num_changing_vars) + 9];
			INaK_ = 2.724*randNums[(blockIdx.x * total_cells * num_changing_vars) + (threadIdx.x * num_changing_vars) + 10];
			kNaCa = 1000 * randNums[(blockIdx.x * total_cells * num_changing_vars) + (threadIdx.x * num_changing_vars) + 11];
			Vleak = 3.6e-4*randNums[(blockIdx.x * total_cells * num_changing_vars) + (threadIdx.x * num_changing_vars) + 12];
			Iup_ = 6.375*randNums[(blockIdx.x * total_cells * num_changing_vars) + (threadIdx.x * num_changing_vars) + 13];
			Vrel = 0.102*randNums[(blockIdx.x * total_cells * num_changing_vars) + (threadIdx.x * num_changing_vars) + 14];
		}

		ENa = RTF*log(Nao / x[cell_num + Nai_i]);
		EK = RTF*log(Ko / x[cell_num + Ki_i]);
		ECa = 0.5*RTF*log(Cao / x[cell_num + Cai_i]);

		EKs = RTF*log((Ko + pKNa*Nao) / (x[cell_num + Ki_i] + pKNa*x[cell_num + Nai_i]));

		INa = GNa_*pow(x[cell_num + m_i], 3)*x[cell_num + h_i] * x[cell_num + j_i] * (x[cell_num + V_i] - ENa);

		ICa = PCa_*x[cell_num + d_i] * x[cell_num + f_i] * x[cell_num + f2_i] * x[cell_num + fCass_i] * 4 * F / RTF*(x[cell_num + V_i] - 15)*(0.25*x[cell_num + Cass_i] * exp(2 * (x[cell_num + V_i] - 15) / RTF) - Cao) / (exp(2 * (x[cell_num + V_i] - 15) / RTF) - 1);

		Ito = Gto_*x[cell_num + r_i] * x[cell_num + s_i] * (x[cell_num + V_i] - EK);

		IKs = GKs_*pow(x[cell_num + xs_i], 2)*(x[cell_num + V_i] - EKs);

		IKr = GKr_*sqrt(Ko / 5400)*x[cell_num + xr1_i] * x[cell_num + xr2_i] * (x[cell_num + V_i] - EK);

		aK1 = 0.1 / (exp(0.06*(x[cell_num + V_i] - EK - 200)) + 1);
		bK1 = (3 * exp(2e-4*(x[cell_num + V_i] - EK + 100)) + exp(0.1*(x[cell_num + V_i] - EK - 10))) / (exp(-0.5*(x[cell_num + V_i] - EK)) + 1);
		xK1 = aK1 / (aK1 + bK1);
		IK1 = GK1_*sqrt(Ko / 5400)*xK1*(x[cell_num + V_i] - EK);

		INCX = kNaCa*pow((pow(KmNa, 3) + pow(Nao, 3)), -1)*pow((KmCa + Cao), -1)*pow((ksat*exp((eta - 1)*x[cell_num + V_i] / RTF) + 1), -1)* (exp(eta*x[cell_num + V_i] / RTF)*pow(x[cell_num + Nai_i], 3)*Cao - exp((eta - 1)*x[cell_num + V_i] / RTF)*pow(Nao, 3)*x[cell_num + Cai_i] * alpha_ncx);

		fNaK = 1 / (1 + 0.1245*exp(-0.1*x[cell_num + V_i] / RTF) + 0.0353*exp(-x[cell_num + V_i] / RTF));
		INaK = INaK_*Ko*x[cell_num + Nai_i] * fNaK / ((Ko + KmKo)*(x[cell_num + Nai_i] + KmNai));

		IpCa = GpCa_*x[cell_num + Cai_i] / (x[cell_num + Cai_i] + KpCa);

		IpK = GpK_*(x[cell_num + V_i] - EK) / (exp(-(x[cell_num + V_i] - 25) / 5.98) + 1);

		INab = GNab_*(x[cell_num + V_i] - ENa);
		ICab = GCab_*(x[cell_num + V_i] - ECa);

		Iion = INa + ICa + Ito + IKs + IKr + IK1 + INCX + INaK + IpCa + IpK + INab + ICab;

		minf = 1 / pow((exp((-56.86 - x[cell_num + V_i]) / 9.03) + 1), 2);
		am = 1 / (exp((-60 - x[cell_num + V_i]) / 5) + 1);
		bm = 0.1 / (exp((x[cell_num + V_i] + 35) / 5) + 1) + 0.1 / (exp((x[cell_num + V_i] - 50) / 200) + 1);
		taum = am*bm;

		hinf = (1 / pow((exp((x[cell_num + V_i] + 71.55) / 7.43) + 1), 2));
		jinf = (1 / pow((exp((x[cell_num + V_i] + 71.55) / 7.43) + 1), 2));
		if (x[cell_num + V_i] >= -40) {
			ah = 0;
			bh = 0.77 / (0.13*(exp((-x[cell_num + V_i] - 10.66) / 11.1) + 1));
			aj = 0;
			bj = 0.6*exp(0.057*x[cell_num + V_i]) / (exp(-0.1*(x[cell_num + V_i] + 32)) + 1);
		}
		else {
			ah = 0.057*exp((-x[cell_num + V_i] - 80) / 6.8);
			bh = 2.7*exp(0.079*x[cell_num + V_i]) + 3.1e5*exp(0.3485*x[cell_num + V_i]);
			aj = (-2.5428e4*exp(0.2444*x[cell_num + V_i]) - 6.948e-6*exp(-0.04391*x[cell_num + V_i]))*(x[cell_num + V_i] + 37.78) / (exp(0.311*(x[cell_num + V_i] + 79.23)) + 1);
			bj = 0.02424*exp(-0.01052*x[cell_num + V_i]) / (exp(-0.1378*(x[cell_num + V_i] + 40.14)) + 1);
		}
		tauh = 1 / (ah + bh);
		tauj = 1 / (aj + bj);

		dinf = 1 / (exp(-(x[cell_num + V_i] + 8) / 7.5) + 1);
		ad = 1.4 / (exp(-(x[cell_num + V_i] + 35) / 13) + 1) + 0.25;
		bd = 1.4 / (exp((x[cell_num + V_i] + 5) / 5) + 1);
		gd = 1 / (exp((50 - x[cell_num + V_i]) / 20) + 1);

		taud = (ad*bd + gd);

		finf = 1 / (exp((x[cell_num + V_i] + 20) / 7) + 1);
		af = 1102.5*exp(-pow((x[cell_num + V_i] + 27), 2) / 225);
		bf = 200 / (1 + exp((13 - x[cell_num + V_i]) / 10));
		gf = 180 / (1 + exp((x[cell_num + V_i] + 30) / 10)) + 20;

		tauf = (af + bf + gf);

		f2inf = 0.67 / (exp((x[cell_num + V_i] + 35) / 7) + 1) + 0.33;
		af2 = 600 * exp(-pow((x[cell_num + V_i] + 25), 2) / 170);
		bf2 = 31 / (1 + exp((25 - x[cell_num + V_i]) / 10));
		gf2 = 16 / (1 + exp((x[cell_num + V_i] + 30) / 10));

		tauf2 = (af2 + bf2 + gf2);

		fCassinf = 0.6 / (1 + pow((x[cell_num + Cass_i] / 50), 2)) + 0.4;

		taufCass = 80 / (1 + pow((x[cell_num + Cass_i] / 50), 2)) + 2;

		rinf = 1 / (exp((20 - x[cell_num + V_i]) / 6) + 1);
		taur = (9.5*exp(-pow((x[cell_num + V_i] + 40), 2) / 1800) + 0.8);
		sinf = 1 / (exp((x[cell_num + V_i] + 20) / 5) + 1);
		taus = (85 * exp(-pow((x[cell_num + V_i] + 45), 2) / 320) + 5 / (exp((x[cell_num + V_i] - 20) / 5) + 1) + 3);

		xsinf = 1 / (exp(-(x[cell_num + V_i] + 5) / 14) + 1);
		axs = 1400 / sqrt(exp(-(x[cell_num + V_i] - 5) / 6) + 1);
		bxs = 1 / (exp((x[cell_num + V_i] - 35) / 15) + 1);
		tauxs = (axs*bxs + 80);

		xr1inf = 1 / (exp(-(x[cell_num + V_i] + 26) / 7) + 1);
		axr1 = 450 / (exp(-(x[cell_num + V_i] + 45) / 10) + 1);
		bxr1 = 6 / (exp((x[cell_num + V_i] + 30) / 11.5) + 1);
		tauxr1 = (axr1*bxr1);

		xr2inf = 1 / (exp((x[cell_num + V_i] + 88) / 24) + 1);
		axr2 = 3 / (exp(-(x[cell_num + V_i] + 60) / 20) + 1);
		bxr2 = 1.12 / (exp((x[cell_num + V_i] - 60) / 20) + 1);
		tauxr2 = (axr2*bxr2);

		Ileak = Vleak*(x[cell_num + CaSR_i] - x[cell_num + Cai_i]);
		Iup = Iup_ / (pow((Kmup / x[cell_num + Cai_i]), 2) + 1);

		kcasr = maxsr - (maxsr - minsr) / (1 + pow((EC_ryr / x[cell_num + CaSR_i]), 2));
		k1_ryr = k1_ryr_prime / kcasr;
		k2_ryr = k2_ryr_prime*kcasr;

		O_ryr = k1_ryr*pow(x[cell_num + Cass_i], 2)*x[cell_num + Rbar_ryr_i] / (k3_ryr + k1_ryr*pow(x[cell_num + Cass_i], 2));
		Irel = Vrel*O_ryr*(x[cell_num + CaSR_i] - x[cell_num + Cass_i]);

		Ixfer = Vxfer*(x[cell_num + Cass_i] - x[cell_num + Cai_i]);

		Bi = pow((1 + Bufc*Kbufc / pow((Kbufc + x[cell_num + Cai_i]), 2)), -1);
		Bss = pow((1 + Bufss*Kbufss / pow((Kbufss + x[cell_num + Cass_i]), 2)), -1);
		BSR = pow((1 + BufSR*KbufSR / pow((KbufSR + x[cell_num + CaSR_i]), 2)), -1);

		ion_current[cell_current_idx] = Iion;

		//new states into temp array
		if (!isinf(x[cell_num + m_i] + step*((minf - x[cell_num + m_i]) / taum)) && !isnan(x[cell_num + m_i] + step*((minf - x[cell_num + m_i]) / taum))) {
			x_temp[cell_num + m_i] = x[cell_num + m_i] + step*((minf - x[cell_num + m_i]) / taum);
		}
		else { x_temp[cell_num + m_i] = x[cell_num + m_i]; }


		if (!isinf(x[cell_num + h_i] + step*((hinf - x[cell_num + h_i]) / tauh)) && !isnan(x[cell_num + h_i] + step*((hinf - x[cell_num + h_i]) / tauh))) {
			x_temp[cell_num + h_i] = x[cell_num + h_i] + step*((hinf - x[cell_num + h_i]) / tauh);
		}
		else { x_temp[cell_num + h_i] = x[cell_num + h_i]; }

		if (!isinf(x[cell_num + j_i] + step*((jinf - x[cell_num + j_i]) / tauj)) && !isnan(x[cell_num + j_i] + step*((jinf - x[cell_num + j_i]) / tauj))) {
			x_temp[cell_num + j_i] = x[cell_num + j_i] + step*((jinf - x[cell_num + j_i]) / tauj);
		}
		else { x_temp[cell_num + j_i] = x[cell_num + j_i]; }

		if (!isinf(x[cell_num + d_i] + step*((dinf - x[cell_num + d_i]) / taud)) && !isnan(x[cell_num + d_i] + step*((dinf - x[cell_num + d_i]) / taud))) {
			x_temp[cell_num + d_i] = x[cell_num + d_i] + step*((dinf - x[cell_num + d_i]) / taud);
		}
		else { x_temp[cell_num + d_i] = x[cell_num + d_i]; }

		if (!isinf(x[cell_num + f_i] + step*((finf - x[cell_num + f_i]) / tauf)) && !isnan(x[cell_num + f_i] + step*((finf - x[cell_num + f_i]) / tauf))) {
			x_temp[cell_num + f_i] = x[cell_num + f_i] + step*((finf - x[cell_num + f_i]) / tauf);
		}
		else { x_temp[cell_num + f_i] = x[cell_num + f_i]; }

		if (!isinf(x[cell_num + f2_i] + step*((f2inf - x[cell_num + f2_i]) / tauf2)) && !isnan(x[cell_num + f2_i] + step*((f2inf - x[cell_num + f2_i]) / tauf2))) {
			x_temp[cell_num + f2_i] = x[cell_num + f2_i] + step*((f2inf - x[cell_num + f2_i]) / tauf2);
		}
		else { x_temp[cell_num + f2_i] = x[cell_num + f2_i]; }

		if (!isinf(x[cell_num + fCass_i] + step*((fCassinf - x[cell_num + fCass_i]) / taufCass)) && !isnan(x[cell_num + fCass_i] + step*((fCassinf - x[cell_num + fCass_i]) / taufCass))) {
			x_temp[cell_num + fCass_i] = x[cell_num + fCass_i] + step*((fCassinf - x[cell_num + fCass_i]) / taufCass);
		}
		else { x_temp[cell_num + fCass_i] = x[cell_num + fCass_i]; }

		if (!isinf(x[cell_num + r_i] + step*((rinf - x[cell_num + r_i]) / taur)) && !isnan(x[cell_num + r_i] + step*((rinf - x[cell_num + r_i]) / taur))) {
			x_temp[cell_num + r_i] = x[cell_num + r_i] + step*((rinf - x[cell_num + r_i]) / taur);
		}
		else { x_temp[cell_num + r_i] = x[cell_num + r_i]; }

		if (!isinf(x[cell_num + s_i] + step*((sinf - x[cell_num + s_i]) / taus)) && !isnan(x[cell_num + s_i] + step*((sinf - x[cell_num + s_i]) / taus))) {
			x_temp[cell_num + s_i] = x[cell_num + s_i] + step*((sinf - x[cell_num + s_i]) / taus);
		}
		else { x_temp[cell_num + s_i] = x[cell_num + s_i]; }

		if (!isinf(x[cell_num + xs_i] + step*((xsinf - x[cell_num + xs_i]) / tauxs)) && !isnan(x[cell_num + xs_i] + step*((xsinf - x[cell_num + xs_i]) / tauxs))) {
			x_temp[cell_num + xs_i] = x[cell_num + xs_i] + step*((xsinf - x[cell_num + xs_i]) / tauxs);
		}
		else { x_temp[cell_num + xs_i] = x[cell_num + xs_i]; }

		if (!isinf(x[cell_num + xr1_i] + step*((xr1inf - x[cell_num + xr1_i]) / tauxr1)) && !isnan(x[cell_num + xr1_i] + step*((xr1inf - x[cell_num + xr1_i]) / tauxr1))) {
			x_temp[cell_num + xr1_i] = x[cell_num + xr1_i] + step*((xr1inf - x[cell_num + xr1_i]) / tauxr1);
		}
		else { x_temp[cell_num + xr1_i] = x[cell_num + xr1_i]; }

		if (!isinf(x[cell_num + xr2_i] + step*((xr2inf - x[cell_num + xr2_i]) / tauxr2)) && !isnan(x[cell_num + xr2_i] + step*((xr2inf - x[cell_num + xr2_i]) / tauxr2))) {
			x_temp[cell_num + xr2_i] = x[cell_num + xr2_i] + step*((xr2inf - x[cell_num + xr2_i]) / tauxr2);
		}
		else { x_temp[cell_num + xr2_i] = x[cell_num + xr2_i]; }

		if (!isinf(x[cell_num + Rbar_ryr_i] + step*(-k2_ryr*x[cell_num + Cass_i] * x[cell_num + Rbar_ryr_i] + k4_ryr*(1 - x[cell_num + Rbar_ryr_i]))) && !isnan(x[cell_num + Rbar_ryr_i] + step*(-k2_ryr*x[cell_num + Cass_i] * x[cell_num + Rbar_ryr_i] + k4_ryr*(1 - x[cell_num + Rbar_ryr_i])))) {
			x_temp[cell_num + Rbar_ryr_i] = x[cell_num + Rbar_ryr_i] + step*(-k2_ryr*x[cell_num + Cass_i] * x[cell_num + Rbar_ryr_i] + k4_ryr*(1 - x[cell_num + Rbar_ryr_i]));
		}
		else { x_temp[cell_num + Rbar_ryr_i] = x[cell_num + Rbar_ryr_i]; }

		if (!isinf(x[cell_num + Cai_i] + step* (Bi*(-(IpCa + ICab - 2 * INCX)*1e6*Acap / (2 * F*Vmyo) + (VSR / Vmyo)*(Ileak - Iup) + Ixfer))) && !isnan(x[cell_num + Cai_i] + step* (Bi*(-(IpCa + ICab - 2 * INCX)*1e6*Acap / (2 * F*Vmyo) + (VSR / Vmyo)*(Ileak - Iup) + Ixfer)))) {
			x_temp[cell_num + Cai_i] = x[cell_num + Cai_i] + step* (Bi*(-(IpCa + ICab - 2 * INCX)*1e6*Acap / (2 * F*Vmyo) + (VSR / Vmyo)*(Ileak - Iup) + Ixfer));
		}
		else { x_temp[cell_num + Cai_i] = x[cell_num + Cai_i]; }

		if (!isinf(x[cell_num + Cass_i] + step*((Bss*(-ICa*1e6*Acap / (2 * F*Vss) + VSR / Vss*Irel - Vmyo / Vss*Ixfer)))) && !isnan(x[cell_num + Cass_i] + step*((Bss*(-ICa*1e6*Acap / (2 * F*Vss) + VSR / Vss*Irel - Vmyo / Vss*Ixfer))))) {
			x_temp[cell_num + Cass_i] = x[cell_num + Cass_i] + step*((Bss*(-ICa*1e6*Acap / (2 * F*Vss) + VSR / Vss*Irel - Vmyo / Vss*Ixfer)));
		}
		else { x_temp[cell_num + Cass_i] = x[cell_num + Cass_i]; }

		if (!isinf(x[cell_num + CaSR_i] + step* (BSR*(Iup - Ileak - Irel))) && !isnan(x[cell_num + CaSR_i] + step* (BSR*(Iup - Ileak - Irel)))) {
			x_temp[cell_num + CaSR_i] = x[cell_num + CaSR_i] + step* (BSR*(Iup - Ileak - Irel));
		}
		else { x_temp[cell_num + CaSR_i] = x[cell_num + CaSR_i]; }

		if (!isinf(x[cell_num + Nai_i] + step*(-(INa + 3 * INCX + 3 * INaK + INab)*1e6*Acap / (F*Vmyo))) && !isnan(x[cell_num + Nai_i] + step*(-(INa + 3 * INCX + 3 * INaK + INab)*1e6*Acap / (F*Vmyo)))) {
			x_temp[cell_num + Nai_i] = x[cell_num + Nai_i] + step*(-(INa + 3 * INCX + 3 * INaK + INab)*1e6*Acap / (F*Vmyo));
		}
		else { x_temp[cell_num + Nai_i] = x[cell_num + Nai_i]; }

		if (!isinf(x[cell_num + Ki_i] + step*(-(Ito + IKs + IKr + IK1 - 2 * INaK + IpK)*1e5*Acap / (F*Vmyo))) && !isnan(x[cell_num + Ki_i] + step*(-(Ito + IKs + IKr + IK1 - 2 * INaK + IpK)*1e5*Acap / (F*Vmyo)))) {
			x_temp[cell_num + Ki_i] = x[cell_num + Ki_i] + step*(-(Ito + IKs + IKr + IK1 - 2 * INaK + IpK)*1e5*Acap / (F*Vmyo));
		}
		else { x_temp[cell_num + Ki_i] = x[cell_num + Ki_i]; }
	}

}

__global__ void updateState(float* x, float* x_temp, int num_cells, int cells_per_thread) {
	int i;
	int idx = cells_per_thread*threadIdx.x;
	int variations = blockIdx.x;

	int limit = idx + cells_per_thread;
	for (; idx<limit; idx++) {
		for (i = 1; i<19; i++) {
			x[(variations * 19 * num_cells) + idx + (i*num_cells)] = x_temp[(variations * 19 * num_cells) + idx + (i*num_cells)];
		}
	}

}

__global__ void compute_voltage(float* x, float* V, float* Iion, float step, float* randNums, int variations, int length, int width, int num_changing_vars, int time, float stimDur, float stimAmp, int tstim, int cells_per_thread, bool local, int* passed, int threshold) {
	bool s2_analysis = false;
	int num_cells = length*width;
	int m;
	int n;
	
	float stim = 0.0;
	float Istim1 = 0.0;
	float Istim2 = 0.0;
	float Vnet_R, Vnet_L, Vnet_U, Vnet_D;
	float rad = 0.0011;
	float deltx = 0.01;
	float rho;
	float Cm = 2;
	float Rmyo;
	float gj;
	int s2_loc = -10000;
	int tstim2;
	int
		idx = cells_per_thread*threadIdx.x;
	int limit = idx + cells_per_thread;

	for (; idx<limit; idx++) {
		m = (blockIdx.x * num_cells) + idx;
		n = (blockIdx.x * num_cells * 19) + idx;
		if (num_changing_vars == 0) {
			gj = 1.27;
			Rmyo = 526;
		}
		else {
			gj = 1.27*randNums[(blockIdx.x * num_cells * num_changing_vars) + (threadIdx.x * num_changing_vars) + 15];
			Rmyo = 526 * randNums[(blockIdx.x * num_cells * num_changing_vars) + (threadIdx.x * num_changing_vars) + 16];
		}

		rho = 3.14159*pow(rad, 2)*(Rmyo + 1000 / gj) / deltx; // total resistivity

		//if (s2_analysis) {
			//tstim2 = s2time[blockIdx.x] / step;
		//}

		if (time%tstim > (stimDur / step))	{ Istim1 = 0.0; }
		else { Istim1 = stimAmp; }

		//if (s2_analysis) {
			//if (time >= tstim2 && time <= (stimDur / step) + tstim2)	{ Istim2 = -150; }
			//else { Istim2 = 0.0; }
		//}

		// Cable Model
		if (width == 1) {
			if (idx == 0) { //first + stimulus
				if (!isinf((x[n]) + (step)*(rad / (2 * rho*Cm*deltx*deltx)*((x[n + 1] - x[n])) - (Iion[m] + Istim1) / Cm)) && !isnan((x[n]) + (step)*(rad / (2 * rho*Cm*deltx*deltx)*((x[n + 1] - x[n])) - (Iion[m] + Istim1) / Cm))) {
					V[m] = (x[n]) + (step)*(rad / (2 * rho*Cm*deltx*deltx)*((x[n + 1] - x[n])) - (Iion[m] + Istim1) / Cm);
					//V[m] = (x[n]) + (step)*( rad/(2*rho*Cm*deltx*deltx)*(x[n+1]-2*x[n] + x[n+length-1]) - (Iion[n]+Istim1) /Cm ) ; // loop
				}
				else { V[m] = x[n]; }
			}
			else if (idx == num_cells - 1){ //last
				if (!isinf((x[n]) + (step)*(rad / (2 * rho*Cm*deltx*deltx)*(-x[n] + x[n - 1]) - (Iion[m]) / Cm)) && !isnan((x[n]) + (step)*(rad / (2 * rho*Cm*deltx*deltx)*(-x[n] + x[n - 1]) - (Iion[m]) / Cm))) {
					V[m] = (x[n]) + (step)*(rad / (2 * rho*Cm*deltx*deltx)*(-x[n] + x[n - 1]) - (Iion[m]) / Cm);
					//V[m] = (x[n]) + (step)*( rad/(2*rho*Cm*deltx*deltx)*(x[n+1-length] - 2*x[n] + x[n-1]) - (Iion[n]) /Cm ); // loop
				}
				else { V[m] = x[n]; }
			}
			else if (idx == 1) { //need stimulus
				if (!isinf((x[n]) + (step)*(rad / (2 * rho*Cm*deltx*deltx)*(x[n + 1] - 2 * x[n] + x[n - 1]) - (Iion[m] + Istim1) / Cm)) && !isnan((x[n]) + (step)*(rad / (2 * rho*Cm*deltx*deltx)*(x[n + 1] - 2 * x[n] + x[n - 1]) - (Iion[m] + Istim1) / Cm))) {
					V[m] = (x[n]) + (step)*(rad / (2 * rho*Cm*deltx*deltx)*(x[n + 1] - 2 * x[n] + x[n - 1]) - (Iion[m] + Istim1) / Cm);
				}
				else { V[m] = x[n]; }
			}
			else if (idx == 2) { //need stimulus
				if (!isinf((x[n]) + (step)*(rad / (2 * rho*Cm*deltx*deltx)*(x[n + 1] - 2 * x[n] + x[n - 1]) - (Iion[m] + Istim1) / Cm)) && !isnan((x[n]) + (step)*(rad / (2 * rho*Cm*deltx*deltx)*(x[n + 1] - 2 * x[n] + x[n - 1]) - (Iion[m] + Istim1) / Cm))) {
					V[m] = (x[n]) + (step)*(rad / (2 * rho*Cm*deltx*deltx)*(x[n + 1] - 2 * x[n] + x[n - 1]) - (Iion[m] + Istim1) / Cm);
				}
				else { V[m] = x[n]; }
			}
			//stim2
			else if (s2_analysis && s2_loc == idx) { //need stimulus
				if (!isinf((x[n]) + (step)*(rad / (2 * rho*Cm*deltx*deltx)*(x[n + 1] - 2 * x[n] + x[n - 1]) - (Iion[m] + Istim2) / Cm)) && !isnan((x[n]) + (step)*(rad / (2 * rho*Cm*deltx*deltx)*(x[n + 1] - 2 * x[n] + x[n - 1]) - (Iion[m] + Istim2) / Cm))) {
					V[m] = (x[n]) + (step)*(rad / (2 * rho*Cm*deltx*deltx)*(x[n + 1] - 2 * x[n] + x[n - 1]) - (Iion[m] + Istim2) / Cm);
				}
				else { V[m] = x[n]; }
			}
			else if (s2_analysis && s2_loc + 1 == idx) { //need stimulus
				if (!isinf((x[n]) + (step)*(rad / (2 * rho*Cm*deltx*deltx)*(x[n + 1] - 2 * x[n] + x[n - 1]) - (Iion[m] + Istim2) / Cm)) && !isnan((x[n]) + (step)*(rad / (2 * rho*Cm*deltx*deltx)*(x[n + 1] - 2 * x[n] + x[n - 1]) - (Iion[m] + Istim2) / Cm))) {
					V[m] = (x[n]) + (step)*(rad / (2 * rho*Cm*deltx*deltx)*(x[n + 1] - 2 * x[n] + x[n - 1]) - (Iion[m] + Istim2) / Cm);
				}
				else { V[m] = x[n]; }
			}
			else if (s2_analysis && s2_loc - 1 == idx) { //need stimulus
				if (!isinf((x[n]) + (step)*(rad / (2 * rho*Cm*deltx*deltx)*(x[n + 1] - 2 * x[n] + x[n - 1]) - (Iion[m] + Istim2) / Cm)) && !isnan((x[n]) + (step)*(rad / (2 * rho*Cm*deltx*deltx)*(x[n + 1] - 2 * x[n] + x[n - 1]) - (Iion[m] + Istim2) / Cm))) {
					V[m] = (x[n]) + (step)*(rad / (2 * rho*Cm*deltx*deltx)*(x[n + 1] - 2 * x[n] + x[n - 1]) - (Iion[m] + Istim2) / Cm);
				}
				else { V[m] = x[n]; }
			}
			else {
				if (!isinf((x[n]) + (step)*(rad / (2 * rho*Cm*deltx*deltx)*((x[n - 1] - x[n]) + (x[n + 1] - x[n])) - (Iion[m]) / Cm)) && !isnan((x[n]) + (step)*(rad / (2 * rho*Cm*deltx*deltx)*((x[n - 1] - x[n]) + (x[n + 1] - x[n])) - (Iion[m]) / Cm))) {
					V[m] = (x[n]) + (step)*(rad / (2 * rho*Cm*deltx*deltx)*((x[n - 1] - x[n]) + (x[n + 1] - x[n])) - (Iion[m]) / Cm);
				}
				else { V[m] = x[n]; }
			}
		}
		//Tissue Model
		else {
			// set which cells will get a stimulus
			if (idx == 0 || idx == 1)					{ stim = Istim1; }
			if (idx == 2 || idx == 0 + length)			{ stim = Istim1; }
			if (idx == 1 + length || idx == 2 + length)			{ stim = Istim1; }
			if (idx == 0 + 2 * length || idx == 1 + 2 * length)			{ stim = Istim1; }
			if (idx == 2 + 2 * length)											{ stim = Istim1; }


			if (threadIdx.x >= 0 && threadIdx.x <= length - 1) { // Top Edge
				Vnet_U = 0.0;
			}
			else {
				Vnet_U = x[n - length] - x[n];
			}


			if (threadIdx.x >= ((width*length) - length) && threadIdx.x <= ((width*length) - 1)) { // Bottom Edge
				Vnet_D = 0.0;
			}
			else {
				Vnet_D = x[n + length] - x[n];
			}


			if (threadIdx.x%length == 0) { // Left Edge
				Vnet_L = 0.0;
				//Vnet_L =  x[n+length-1] - x[n]; // tissue loop
			}
			else {
				Vnet_L = x[n - 1] - x[n];
			}

			if (threadIdx.x%length == (length - 1)) { // Right Edge
				Vnet_R = 0.0;
				//Vnet_R = x[n+1-length] - x[n]; // tissue loop
			}
			else {
				Vnet_R = x[n + 1] - x[n];
			}

			if (!isinf((x[n]) + (step)*(rad / (2 * rho*Cm*deltx*deltx)*(Vnet_R + Vnet_L + Vnet_U + Vnet_D) - (Iion[m] + stim) / Cm)) && !isnan((x[n]) + (step)*(rad / (2 * rho*Cm*deltx*deltx)*(Vnet_R + Vnet_L + Vnet_U + Vnet_D) - (Iion[m] + stim) / Cm))) {
				V[m] = (x[n]) + (step)*(rad / (2 * rho*Cm*deltx*deltx)*(Vnet_R + Vnet_L + Vnet_U + Vnet_D) - (Iion[m] + stim) / Cm);
			}
			else { V[m] = x[n]; }
		}
		if(local && V[m] >= threshold && passed[m] == 0)
		{
			passed[m] = time;
		}
	}

}

__global__ void update_voltage(float* x, float* V, int total_cells, int cells_per_thread) {
	int idx = cells_per_thread*threadIdx.x;

	int limit = idx + cells_per_thread;
	for (; idx<limit; idx++) {
		int m = (blockIdx.x * total_cells) + idx;
		int n = (blockIdx.x * total_cells * 19) + idx;

		x[n] = V[m];
	}
}

__global__ void computeVelocity(float* voltage, int iterations, int num_cells, float* vel, float time, int length, int width) {
	float startTP = 0.0;
	float endTP = 0.0;
	float deltx = 0.01;
	int i, k;
	int idx = threadIdx.x;
	float distance;

	int start = idx*num_cells*iterations;

	for (i = 0; i<iterations; i++) { // Looking at first cell voltage only
		if (voltage[start + i] >= -55) {
			startTP = (i - start)*time;
			break;
		}
	}

	for (k = (iterations*(num_cells - 1)); k<(iterations*num_cells); k++) { // Looking at last cell voltage only
		if (voltage[start + k] >= -55) {
			endTP = ((k - start) - (iterations*(num_cells - 1)))*time;
			break;
		}
	}

	//distance = ;

	vel[idx] = endTP - startTP;

}

__global__ void computeLocal(int* passed, int num_cells, int* vel)
{
	int idx = blockIdx.x * (num_cells -1 ) + threadIdx.x;
	int idxP = blockIdx.x * num_cells + threadIdx.x;
	vel[idx] = passed[idxP + 1] - passed[idxP];
}

__global__ void init_randomNums(curandState *state, int sf) {
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	curand_init(sf + blockIdx.x + threadIdx.x, idx, blockDim.x + threadIdx.x, &state[idx]);
}

__global__ void make_randomNums( float* randArray, int num_cells, int num_changing_vars, int sf) {
	curandState local;
	int idx = num_cells*blockIdx.x*num_changing_vars + threadIdx.x;
	float sigma1D = 0.15; //SD of variation of simulation
	float sigma2D = 0; //SD of variation of cells within cable
	int i;
	

	curand_init(sf, idx, 0, &local);

	randArray[idx] = exp(sigma1D*curand_normal(&local)); //computes randNum with wider distribution for first cell

	for (i = (idx + num_changing_vars); i < (idx + (num_changing_vars * num_cells)); i += num_changing_vars)
	{
		// computes randNums to create a smaller norm distribution for rest of cells in cable
		randArray[i] = randArray[idx] + (curand_normal(&local) * sigma2D);
		//randArray[i] = randArray[idx];
	}
}

__global__ void initialize_time_s2(float begin_time, float interval, float* time_s2) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	time_s2[idx] = begin_time + (interval*idx);
}

__global__ void percentage_excited(float* V_array, int iterations, int num_cells, float* percent, int variations) {
	int idx = threadIdx.x;
	int last_row = iterations - 1;
	int start = last_row + (idx*num_cells*iterations);
	int num_excited = 0;
	int i;

	for (i = 0; i<num_cells; i++) {
		if (V_array[start + (i*iterations)] >= -55) {
			num_excited++;
		}
	}
	percent[idx] = ((float)num_excited / (float)num_cells);
}

int main(int argc, const char* argv[])
{
	int i, ii, iii;
	int time = 0;
	FILE *fV = fopen("tt06_GPU_Voltage.txt", "w");
	FILE *ft = fopen("tt06_GPU_Time.txt", "w");
	FILE *params = fopen("tt06_GPU_SA_parameters.txt", "w");
	FILE *allparams = fopen("tt06_GPU_2D_all_parameters.txt", "w");
	FILE *output = fopen("tt06_GPU_SA_output.txt", "w");
	FILE *s2output = fopen("tt06_GPU_s2_Analysis.txt", "w");
	FILE* p = fopen("tt06_GPU_passed_threshold.txt", "w");
	FILE* v = fopen("tt06_GPU_local_velocity.txt","w");
	int index = 0;
	float* host_vars;
	float* dev_vars;
	float* dev_ion_currents;
	float* dev_x_temp;
	float* host_Vtemp;
	float* dev_Vtemp;
	float* V_array;
	float* t_array;
	float* dev_V_array;
	float* dev_vel;
	float* vel;
	float* s2_times;
	float* s2_times_dev;
	float* percent_excited;
	float* dev_percent_excited;
	cudaEvent_t start, stop;
	float elapsedTime;
	float* dev_randNums;
	float* randNums;
	int size;
	float begin_time;
	float end_time;
	float test_interval;
	int total_s2_times;
	int s2_loc;
	int seed_factor;

	//Number of Parameters in the Model
	int num_param = 19;

	// Assume only running 1 simulation initially
	int simulations = 500;

	// Time Step Variables
	float step = 0.002;
	float tend = 1000;
	int iterations = tend / step; // 50/.002
	float skip_time_value = 0.5; //ms 
	int skip_timept = skip_time_value / step; // skipping time points in voltage array & time array
	int total_timepts = iterations / skip_timept + 1;

	// Number of Cells
	int length = 100;
	int width = 1;
	int num_cells = length*width;
	int cells_per_thread = 1; // for cell numbers > 500, one thread may need to work on more than one cell

	//Stimulus Variables
	float stimDur = 2.0;
	float stimAmp = -60.0;
	float stimInterval = 1000;
	int tstim = stimInterval / step;

	// Sensitivity Analysis?
	int num_changing_vars = 17;

	int* dev_passed;
	int* dev_velLocal;
	int* host_passed;
	int* host_velLocal;
	int threshold = -55;
	bool local = true;


	//int num_changing_vars = 0;

	// S2 Analysis?
	bool s2_analysis = false;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	if (s2_analysis) {
		begin_time = 900;
		end_time = 930;
		test_interval = 10;
		s2_loc = 5;
		total_s2_times = (end_time - begin_time) / test_interval; //make sure not too many threads
		simulations = total_s2_times;
		s2_times = (float*)malloc(sizeof(float)*total_s2_times);
		cudaMalloc(&s2_times_dev, sizeof(float)*total_s2_times);
		initialize_time_s2 << <1, total_s2_times >> >(begin_time, test_interval, s2_times_dev);
		cudaMemcpy(s2_times, s2_times_dev, total_s2_times*sizeof(float), cudaMemcpyDeviceToHost);
	}
	else
	{
		s2_loc = 0;
		s2_analysis = 0;
		total_s2_times = 0;
		s2_times = 0;
		cudaMalloc(&s2_times_dev, sizeof(float)*2);
	}
	size = num_param*num_cells*simulations;

	if (num_changing_vars != 0) {
		printf("Enter a seed factor\n");
		scanf("%d", &seed_factor);
		simulations = 500;
		//cudaMalloc(&rndState, simulations*num_changing_vars * sizeof(curandState));
		//cudaMalloc(&rndState, simulations*num_changing_vars);
		cudaMalloc(&dev_randNums, sizeof(float)*simulations*num_changing_vars * num_cells);
		randNums = (float*)malloc(sizeof(float)*simulations*num_changing_vars * num_cells);
		//init_randomNums << <simulations, num_changing_vars >> >(rndState, seed_factor);
		//cudaDeviceSynchronize();
		make_randomNums <<<simulations, num_changing_vars >>>(dev_randNums, num_cells, num_changing_vars, seed_factor);

	}
	size = num_param*num_cells*simulations;

	// vars array contains voltage&state vartiables for all cells across all simulations
	host_vars = (float *)malloc(sizeof(float)*size);
	cudaMalloc(&dev_vars, sizeof(float)*size);

	// results of the computeState kernel
	cudaMalloc(&dev_ion_currents, sizeof(float)*num_cells*simulations);
	cudaMalloc(&dev_x_temp, sizeof(float)*size);

	// result of the computeVoltage kernel 
	host_Vtemp = (float*)malloc(sizeof(float)*num_cells*simulations);
	cudaMalloc(&dev_Vtemp, sizeof(float)*num_cells*simulations);

	V_array = (float*)malloc(sizeof(float)*(total_timepts*num_cells*simulations));
	t_array = (float*)malloc(sizeof(float)*(total_timepts*simulations));

	fprintf(fV, "V = [ \n");


	host_velLocal = (int*)malloc(simulations * (num_cells-1) *sizeof(int));
	host_passed = (int*)malloc(simulations*num_cells*sizeof(int));
	cudaMalloc(&dev_velLocal, sizeof(int)* simulations*(num_cells-1));
	cudaMalloc(&dev_passed, sizeof(int)*simulations*num_cells);

	// Initialize vars array with initial conditions
	initialConditions <<<simulations, (num_cells / cells_per_thread) >>>(dev_vars, num_param, num_cells, cells_per_thread, dev_passed);

	while (time<iterations) {

		computeState <<<simulations, (num_cells / cells_per_thread) >>>(dev_vars, dev_ion_currents, num_cells, step, dev_randNums, simulations, dev_x_temp, num_changing_vars, cells_per_thread);
		updateState <<<simulations, (num_cells / cells_per_thread) >>>(dev_vars, dev_x_temp, num_cells, cells_per_thread);

		compute_voltage <<<simulations, (num_cells / cells_per_thread) >>>(dev_vars, dev_Vtemp, dev_ion_currents, step, dev_randNums, simulations, length, width, num_changing_vars, time, stimDur, stimAmp, tstim, cells_per_thread, local, dev_passed, threshold);
		update_voltage <<<simulations, (num_cells / cells_per_thread) >>>(dev_vars, dev_Vtemp, num_cells, cells_per_thread);

		//update Voltage and time arrays and write data to file
		if (index > total_timepts)
			printf("Error\n");
		if (time%skip_timept == 0) {
			cudaMemcpy(host_Vtemp, dev_Vtemp, num_cells*simulations*sizeof(float), cudaMemcpyDeviceToHost);
			for (i = 0; i<num_cells*simulations; i++) {
				V_array[i * total_timepts + index] = host_Vtemp[i];
				fprintf(fV, "%f\t ", host_Vtemp[i]);
			}
			fprintf(fV, "\n");
			fprintf(ft, "%f \n", time*step);
			for (i = 0; i<simulations; i++) {
				t_array[(index*simulations) + i] = time*step;
			}
			index++;
		}
		time++;
	}
	cudaDeviceSynchronize();
	fprintf(fV, "]; \n");


	/*
	The Model Computations are Finished
	This last section of code is only writing data to file(s) and cleaning up the memory
	*/

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

	free(host_vars);
	cudaFree(dev_vars);
	cudaFree(dev_ion_currents);
	cudaFree(dev_x_temp);
	cudaFree(dev_Vtemp);

	printf("Elapsed Time = %f s \n", elapsedTime / 1000);
	printf("\n");
	printf("Calculating Simulation outputs...\n");
	printf("\n");

	if (num_changing_vars != 0) {
		vel = (float*)malloc(sizeof(float)*simulations);
		cudaMalloc(&dev_vel, (sizeof(float)*simulations));

		cudaMalloc(&dev_V_array, sizeof(float)*(total_timepts*num_cells*simulations));
		cudaMemcpy(dev_V_array, V_array, sizeof(float)*(total_timepts*num_cells*simulations), cudaMemcpyHostToDevice);

		computeVelocity <<<1, simulations >>>(dev_V_array, total_timepts, num_cells, dev_vel, step*skip_timept, length, width);
		computeLocal <<<simulations,(num_cells-1)>>>(dev_passed,num_cells,dev_velLocal);


		cudaMemcpy(vel, dev_vel, simulations*sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(randNums, dev_randNums, num_changing_vars*simulations*sizeof(float) * num_cells, cudaMemcpyDeviceToHost);
		cudaMemcpy(host_passed,dev_passed, sizeof(int)*simulations*num_cells, cudaMemcpyDeviceToHost);
		cudaMemcpy(host_velLocal,dev_velLocal,sizeof(int)* simulations*(num_cells-1), cudaMemcpyDeviceToHost);	

		fprintf(params, "A = [ \n");
		fprintf(allparams, "A = [ \n");
		for (i = 0; i<simulations; i++) {
			for (ii = 0; ii<num_changing_vars; ii++) {
				//printf( "%f\t", randNums[(i*num_changing_vars * num_cells)+ii]);
				fprintf(params, "%f\t", randNums[(i*num_changing_vars * num_cells) + ii]);
			}
			for (iii = 0; iii< num_cells; iii++){
				for (ii = 0; ii<num_changing_vars; ii++) {
					//printf( "%f\t", randNums[(i*num_changing_vars * num_cells)+ii]);
					fprintf(allparams, "%f\t", randNums[(i* num_cells * num_changing_vars) + (iii * num_changing_vars) + ii]);
				}
				fprintf(allparams, "\n");
			}
			for(ii = 0; ii <  num_cells; ii++)
			{
				fprintf(p, "%i\t", host_passed[i * num_cells + ii] );
			}
			for(ii = 0; ii < num_cells-1;ii++)
			{
				fprintf(v, "%i\t", host_velLocal[i * (num_cells-1) + ii]);
			}
			fprintf(p, "\n" );
			fprintf(v, "\n" );
			fprintf(params, "\n");

		}
		fprintf(params, "]; \n");
		//fprintf(output, "\n");
		fprintf(output, "Vel = [ \n");
		for (i = 0; i<simulations; i++) {
			fprintf(output, "%f\n", vel[i]);
		}
		fprintf(output, "]; \n");
		//cudaFree(rndState);
		cudaFree(dev_randNums);
		free(randNums);
		free(vel);
		cudaFree(dev_vel);
		cudaFree(dev_velLocal);
		cudaFree(dev_passed);
		free(host_velLocal);
		free(host_passed);

	}

	if (s2_analysis) {
		cudaMalloc(&dev_V_array, sizeof(float)*(total_timepts*num_cells*simulations));
		cudaMemcpy(dev_V_array, V_array, sizeof(float)*(total_timepts*num_cells*simulations), cudaMemcpyHostToDevice);

		percent_excited = (float*)malloc(sizeof(float)*total_s2_times);
		cudaMalloc(&dev_percent_excited, sizeof(float)*total_s2_times);

		percentage_excited << <1, total_s2_times >> >(dev_V_array, total_timepts, num_cells, dev_percent_excited, simulations);

		cudaMemcpy(percent_excited, dev_percent_excited, total_s2_times*sizeof(float), cudaMemcpyDeviceToHost);

		fprintf(s2output, "A = [ \n");
		for (i = 0; i<simulations; i++) {
			fprintf(s2output, "%f\n", s2_times[i]);
		}
		fprintf(s2output, "]; \n");
		fprintf(s2output, "\n");
		fprintf(s2output, "% = [ \n");
		for (i = 0; i<simulations; i++) {
			fprintf(s2output, "%f\n", percent_excited[i]);
		}
		fprintf(s2output, "]; \n");

		free(percent_excited);
		cudaFree(dev_percent_excited);
	}
	free(V_array);
	cudaFree(dev_V_array);
	//free(host_Vtemp); //Makes sure this works
	printf("Program is Done\n");
}
