package main

import (
	"image/color"
)

var Y []float64 = []float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2}

var CMap map[float64]color.Color = map[float64]color.Color{
	0: color.RGBA{0, 0, 255, 255},
	1: color.RGBA{255, 0, 0, 255},
	2: color.RGBA{0, 255, 0, 255},
}

var X [][]float64 = [][]float64{
	{0.00000000e+00, 0.00000000e+00},
	{2.99555692e-03, 9.64660756e-03},
	{1.28809698e-02, 1.55628482e-02},
	{2.99747903e-02, 4.44809627e-03},
	{3.93124595e-02, 9.32828337e-03},
	{8.28829070e-04, 5.04982509e-02},
	{5.34835160e-02, 2.85062827e-02},
	{4.17361967e-02, 5.70752136e-02},
	{5.54633923e-02, 5.87686822e-02},
	{8.16038325e-02, 4.00659069e-02},
	{8.91875103e-02, 4.74197045e-02},
	{1.07160836e-01, -2.93638241e-02},
	{1.21183202e-01, -2.64751446e-03},
	{1.28777727e-01, 2.56794747e-02},
	{1.41112968e-01, -9.22449492e-03},
	{1.50579467e-01, -1.68126337e-02},
	{1.13476366e-01, -1.15077794e-01},
	{1.71552509e-01, -7.51816481e-03},
	{1.67186841e-01, -7.14591444e-02},
	{1.91325873e-01, 1.50793232e-02},
	{1.36771902e-01, 1.48679554e-01},
	{1.35606423e-01, -1.63114399e-01},
	{1.04024746e-01, -1.96370989e-01},
	{2.15633541e-01, -8.64653736e-02},
	{-9.83033329e-02, -2.21598670e-01},
	{2.46031418e-01, -5.68994507e-02},
	{1.24166250e-01, -2.31420174e-01},
	{1.32641941e-01, -2.38298729e-01},
	{-1.23804301e-01, -2.54291832e-01},
	{-1.46155268e-01, -2.53862590e-01},
	{1.02452915e-02, -3.02857071e-01},
	{-5.56905754e-02, -3.08139235e-01},
	{1.11795291e-01, -3.03283602e-01},
	{2.38806710e-01, -2.32556373e-01},
	{-4.07022871e-02, -3.41013879e-01},
	{-1.60721049e-01, -3.14890444e-01},
	{-3.25695068e-01, -1.61722347e-01},
	{-3.47926050e-01, -1.36481121e-01},
	{-1.85074612e-01, -3.36272657e-01},
	{-2.37354815e-01, -3.14405680e-01},
	{-1.47814021e-01, -3.76031458e-01},
	{-1.18350260e-01, -3.96870673e-01},
	{-1.04012094e-01, -4.11294430e-01},
	{-3.56893986e-01, 2.47549772e-01},
	{-3.85084093e-01, -2.21903399e-01},
	{-4.21105951e-01, -1.71117917e-01},
	{-3.57668996e-01, -2.96596050e-01},
	{-4.32741880e-01, 1.95242479e-01},
	{-3.79777759e-01, -3.01408201e-01},
	{-4.90725726e-01, 6.45234585e-02},
	{-5.02019227e-01, -5.52514270e-02},
	{-4.15472955e-01, 3.04570675e-01},
	{-5.04104018e-01, 1.47544429e-01},
	{-5.34662068e-01, 2.72004325e-02},
	{-4.07161713e-01, 3.62960041e-01},
	{-2.72887409e-01, 4.83915746e-01},
	{-3.16287279e-01, 4.68966633e-01},
	{-2.10548550e-01, 5.35878778e-01},
	{-3.95798415e-01, 4.31942016e-01},
	{-2.88308799e-01, 5.21580160e-01},
	{-3.21319461e-01, 5.13870835e-01},
	{-1.82880193e-01, 5.88396072e-01},
	{-2.59437025e-01, 5.69997609e-01},
	{-4.48756903e-01, 4.51193869e-01},
	{1.72540992e-01, 6.23013735e-01},
	{5.35037480e-02, 6.54381990e-01},
	{-2.78893858e-01, 6.05526745e-01},
	{4.44151521e-01, 5.10630906e-01},
	{9.03030708e-02, 6.80906713e-01},
	{4.55620110e-01, 5.27424932e-01},
	{6.45984292e-01, 2.87494838e-01},
	{5.84617138e-01, 4.15401071e-01},
	{7.27224171e-01, 8.40271171e-03},
	{3.36008549e-01, 6.56367481e-01},
	{7.35656083e-01, 1.32395625e-01},
	{6.16297543e-01, 4.40565974e-01},
	{6.28073633e-01, 4.41419423e-01},
	{7.26366043e-01, 2.78083950e-01},
	{7.81131148e-01, 1.02893755e-01},
	{7.88548410e-01, -1.22324094e-01},
	{7.57483721e-01, 2.81448126e-01},
	{5.82662702e-01, -5.74391544e-01},
	{6.53483987e-01, -5.08931339e-01},
	{8.14698577e-01, 1.97872922e-01},
	{1.65076673e-01, -8.32271755e-01},
	{-9.34221819e-02, -8.53488147e-01},
	{1.28543124e-01, -8.59123707e-01},
	{5.83978891e-01, -6.56686246e-01},
	{7.80273199e-01, -4.25790071e-01},
	{-8.29009637e-02, -8.95159364e-01},
	{4.63849485e-01, -7.81850338e-01},
	{-3.39549899e-01, -8.54177773e-01},
	{2.57485043e-02, -9.28936124e-01},
	{-4.14920419e-01, -8.42794180e-01},
	{-2.33394891e-01, -9.20362711e-01},
	{-4.80520338e-01, -8.30617011e-01},
	{-2.65596241e-01, -9.32615042e-01},
	{-9.34677541e-01, -2.93908089e-01},
	{-5.06961346e-01, -8.50229502e-01},
	{-7.00584829e-01, -7.13569164e-01},
	{-0.00000000e+00, -0.00000000e+00},
	{-2.37636777e-05, -1.01009822e-02},
	{-2.85904994e-03, -1.99986864e-02},
	{-2.96508409e-02, -6.25310186e-03},
	{-1.54620195e-02, -3.73284407e-02},
	{-4.47676741e-02, 2.33798083e-02},
	{-5.01458272e-02, -3.40366103e-02},
	{-5.57611808e-02, -4.34762090e-02},
	{-5.77887744e-02, 5.64836562e-02},
	{-7.21631050e-02, 5.52896857e-02},
	{-5.88949509e-02, 8.20635408e-02},
	{-9.36308503e-02, 5.98242730e-02},
	{-1.18445486e-01, -2.57496666e-02},
	{-3.85806710e-02, 1.25517607e-01},
	{-1.35726988e-01, 3.97006609e-02},
	{-9.16938111e-02, 1.20619588e-01},
	{-7.41407052e-02, 1.43606886e-01},
	{-1.37312800e-01, 1.03111513e-01},
	{-7.76325166e-02, 1.64411202e-01},
	{-3.55057046e-02, 1.88606262e-01},
	{-7.07798749e-02, 1.89215153e-01},
	{-1.77902266e-01, 1.15525723e-01},
	{-4.30656187e-02, 2.18009338e-01},
	{9.47438180e-02, 2.12126598e-01},
	{-1.13965355e-01, 2.13965908e-01},
	{-2.92110667e-02, 2.50830054e-01},
	{-4.12407555e-02, 2.59368002e-01},
	{2.41241619e-01, 1.27211019e-01},
	{1.59243613e-01, 2.33737692e-01},
	{1.57247782e-01, 2.47145116e-01},
	{2.39067562e-02, 3.02085787e-01},
	{2.32032150e-01, 2.10267216e-01},
	{1.04369961e-01, 3.05918366e-01},
	{2.35107496e-01, 2.36295521e-01},
	{1.79523319e-01, 2.92777270e-01},
	{3.41323912e-01, 9.21153203e-02},
	{3.55456382e-01, 7.66952783e-02},
	{3.27276438e-01, 1.80470958e-01},
	{3.81890804e-01, 3.86176892e-02},
	{2.89765954e-01, 2.66878128e-01},
	{2.68961608e-01, 3.01510036e-01},
	{4.03800040e-01, -9.19708386e-02},
	{4.16720092e-01, -7.95361474e-02},
	{3.76112401e-01, -2.17241064e-01},
	{3.19485776e-02, -4.43294644e-01},
	{2.89429367e-01, -3.50488544e-01},
	{4.63996798e-01, -2.45621055e-02},
	{1.85740978e-01, -4.36904401e-01},
	{4.84158278e-01, -2.58612800e-02},
	{4.13589716e-01, -2.71879643e-01},
	{3.22098076e-01, -3.89010072e-01},
	{-1.52347535e-01, -4.92109060e-01},
	{3.86825025e-01, -3.55326086e-01},
	{3.71773154e-01, -3.85211796e-01},
	{1.62570089e-01, -5.20664632e-01},
	{2.58585244e-01, -4.91706878e-01},
	{-2.77088165e-01, -4.93142456e-01},
	{2.71145850e-01, -5.07914066e-01},
	{2.40553409e-01, -5.34195065e-01},
	{-1.94979012e-02, -5.95640540e-01},
	{-6.26916364e-02, -6.02809429e-01},
	{-6.09022141e-01, -9.35261697e-02},
	{-5.36692202e-01, -3.22748154e-01},
	{-4.09688324e-01, -4.86943692e-01},
	{-9.27474126e-02, -6.39776886e-01},
	{-6.28976762e-01, -1.88326120e-01},
	{-2.95126259e-01, -5.97783387e-01},
	{-1.91789702e-01, -6.49023235e-01},
	{-6.75281405e-01, 1.25632852e-01},
	{-6.89081788e-01, -1.04561344e-01},
	{-6.84671879e-01, 1.76559985e-01},
	{-7.16190279e-01, 3.75063084e-02},
	{-6.63356125e-01, 2.98134685e-01},
	{-7.17866004e-01, -1.68488741e-01},
	{-7.09262490e-01, -2.35934705e-01},
	{-6.08644307e-01, 4.51079965e-01},
	{-7.67671525e-01, -2.83790333e-03},
	{-7.68544078e-01, 1.19491868e-01},
	{-7.33914554e-01, 2.86570013e-01},
	{-6.05272949e-01, 5.20015776e-01},
	{-6.55299366e-01, 4.72839653e-01},
	{-7.99242437e-01, 1.75022930e-01},
	{-6.42871678e-01, 5.22272348e-01},
	{-8.38365436e-01, -5.55246091e-03},
	{-4.41790260e-02, 8.47333908e-01},
	{-7.50967443e-01, 4.16194111e-01},
	{-5.69446802e-01, 6.56008542e-01},
	{-3.12734917e-02, 8.78231227e-01},
	{-2.88415402e-01, 8.40797246e-01},
	{7.10775614e-01, 5.50436974e-01},
	{-3.58261794e-01, 8.35520685e-01},
	{4.21108097e-01, 8.17056775e-01},
	{3.82092923e-01, 8.47107053e-01},
	{-3.72496545e-02, 9.38655138e-01},
	{7.47896850e-01, 5.84970891e-01},
	{5.88822305e-01, 7.57702231e-01},
	{9.14996028e-01, 3.21083277e-01},
	{9.58145797e-01, 2.04843029e-01},
	{8.38562548e-01, -5.26035070e-01},
	{9.69426990e-01, -2.45380044e-01},
	{0.00000000e+00, 0.00000000e+00},
	{9.14306752e-03, 4.29356797e-03},
	{1.91021413e-02, -6.57494133e-03},
	{2.96353418e-02, -6.32615155e-03},
	{3.85543592e-02, -1.20850252e-02},
	{3.78438421e-02, 3.34455334e-02},
	{5.96956834e-02, -1.04651796e-02},
	{7.04677626e-02, 5.81241259e-03},
	{6.98159263e-02, -4.06900719e-02},
	{8.22631791e-02, -3.86940874e-02},
	{5.07112965e-02, -8.73579159e-02},
	{7.33842030e-02, -8.34292397e-02},
	{4.56192940e-02, -1.12299860e-01},
	{1.03771001e-01, -8.04656297e-02},
	{1.22611716e-01, -7.04579800e-02},
	{9.53920186e-02, -1.17716625e-01},
	{6.04712553e-02, -1.49876654e-01},
	{1.44758979e-02, -1.71105921e-01},
	{-1.48597717e-01, -1.04769118e-01},
	{1.78390555e-02, -1.91088319e-01},
	{8.76751468e-02, -1.82003379e-01},
	{9.10300482e-03, -2.11925805e-01},
	{1.60965777e-04, -2.22222164e-01},
	{-1.25921011e-01, -1.95238277e-01},
	{8.02290589e-02, -2.28763655e-01},
	{-1.35362729e-01, -2.13180527e-01},
	{-1.71858355e-01, -1.98588163e-01},
	{-2.05292791e-01, -1.79541185e-01},
	{-1.45042375e-01, -2.42805570e-01},
	{-2.14010790e-01, -2.00017393e-01},
	{-9.73699614e-02, -2.86960721e-01},
	{-2.44014725e-01, -1.96234643e-01},
	{-2.66582936e-01, -1.82791352e-01},
	{-3.33262712e-01, -6.86169323e-03},
	{-2.61409342e-01, -2.22738206e-01},
	{-3.30370843e-01, 1.25866413e-01},
	{-2.47638479e-01, 2.66282916e-01},
	{-2.43963838e-01, -2.83127666e-01},
	{-3.38751167e-01, 1.80498064e-01},
	{-2.97227621e-01, 2.58542061e-01},
	{-3.98896396e-01, 6.42675310e-02},
	{-3.84470344e-01, 1.53933972e-01},
	{-3.46745014e-01, 2.44437188e-01},
	{-3.50296497e-01, 2.56800681e-01},
	{-3.31782788e-01, 2.95721233e-01},
	{-4.49742019e-01, 6.59067407e-02},
	{4.17104848e-02, 4.62770551e-01},
	{7.30770975e-02, 4.69089448e-01},
	{-3.08974892e-01, 3.73647630e-01},
	{-3.92718226e-01, 3.01243097e-01},
	{8.95722732e-02, 4.97044086e-01},
	{-1.36970177e-01, 4.96608764e-01},
	{9.95909050e-02, 5.15724599e-01},
	{3.27057391e-02, 5.34353554e-01},
	{3.37681264e-01, 4.28359687e-01},
	{3.86817127e-01, 3.98766190e-01},
	{9.06921476e-02, 5.58338881e-01},
	{-3.84592935e-02, 5.74471653e-01},
	{-3.84191386e-02, 5.84597528e-01},
	{5.39511859e-01, 2.53169537e-01},
	{1.98358148e-01, 5.72680950e-01},
	{4.31343615e-01, 4.39997524e-01},
	{4.62868482e-01, 4.21850264e-01},
	{5.94172835e-01, 2.27853715e-01},
	{2.30009332e-01, 6.04162455e-01},
	{6.55944824e-01, 2.85461489e-02},
	{6.49851620e-01, -1.48786068e-01},
	{6.72798991e-01, -7.31845647e-02},
	{6.86836302e-01, -6.66679395e-03},
	{6.62706614e-01, -2.15839580e-01},
	{5.96542180e-01, -3.79587144e-01},
	{4.35550809e-01, 5.69763780e-01},
	{1.71986911e-02, -7.27069318e-01},
	{4.78151858e-01, -5.61329544e-01},
	{6.74676895e-01, -3.21760178e-01},
	{5.90618432e-01, -4.74437416e-01},
	{2.63736039e-01, -7.20951378e-01},
	{4.09585685e-01, -6.61194265e-01},
	{7.77826965e-01, -1.25451937e-01},
	{-5.36717594e-01, -5.90513289e-01},
	{1.98979661e-01, -7.83199668e-01},
	{-3.31806332e-01, -7.47881055e-01},
	{2.74292469e-01, -7.81547248e-01},
	{-6.44940972e-01, -5.35666525e-01},
	{-2.93879867e-01, -7.95965552e-01},
	{-4.96032059e-01, -7.00800896e-01},
	{9.54423249e-02, -8.63427818e-01},
	{-7.90659189e-01, -3.83570313e-01},
	{-7.27896988e-01, -5.10185719e-01},
	{-8.80292356e-01, -1.82395741e-01},
	{-4.37859565e-01, -7.96696484e-01},
	{-5.75886071e-01, -7.16427982e-01},
	{-7.67105818e-01, 5.24532199e-01},
	{-5.22328019e-01, -7.80790865e-01},
	{-8.67685974e-01, -3.85566771e-01},
	{-9.11101043e-01, 3.01196396e-01},
	{-9.64920223e-01, -9.61308330e-02},
	{-9.50698435e-01, 2.37015888e-01},
	{-9.79386806e-01, -1.43879965e-01},
	{-9.42788780e-01, 3.33390683e-01},
}
