# CoupledConditionalSMC.jl

[![Build Status](https://travis-ci.org/awllee/CoupledConditionalSMC.jl.svg?branch=master)](https://travis-ci.org/awllee/CoupledConditionalSMC.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/axdu0a4bg7s7ilpb/branch/master?svg=true)](https://ci.appveyor.com/project/awllee/coupledconditionalsmc-jl/branch/master)
[![Coverage Status](https://coveralls.io/repos/github/awllee/CoupledConditionalSMC.jl/badge.svg?branch=master)](https://coveralls.io/github/awllee/CoupledConditionalSMC.jl?branch=master)
[![codecov.io](http://codecov.io/github/awllee/CoupledConditionalSMC.jl/coverage.svg?branch=master)](http://codecov.io/github/awllee/CoupledConditionalSMC.jl?branch=master)

This package provides an implementation of a simple coupled conditional SMC
algorithm with index-coupled backward sampling, as described in

Lee, A., Singh, S.S. and Vihola, M., 2018. [Coupled conditional backward sampling particle filter](https://arxiv.org/abs/1806.05852). arXiv:1806.05852.

This package runs on Julia v1.0.

In the future it should be easier to package the code up. For the present, to use the code, you should run

```julia
using Pkg
Pkg.add(PackageSpec(url="https://github.com/awllee/SimpleSMC.jl.git"))
Pkg.add(PackageSpec(url="https://github.com/awllee/CoupledConditionalSMC.jl.git"))
```

To run the demos you should first run
```julia
using Pkg
Pkg.add("SequentialMonteCarlo")
Pkg.add("SMCExamples")
Pkg.add("RNGPool")
```

and then either play with demo code in the REPL or Juno, or run all the demos from the `demo/` folder using
```
sh runDemos.sh
```
which will place output in the `demo/output` directory.
