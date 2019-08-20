using JLD2
using FileIO

function test_merge(fname_base="output/simpleDemoCompareCost_results_", files=100)
    if files < 1
        return
    end
    o = load(string(fname_base, 1, ".jld2"))
    AS_cost_ = o["AS_cost"]
    BS_cost_ = o["BS_cost"]
    ts_ = o["ts"]
    Ns_ = o["Ns"]
    maxIter_ = o["maxIter"]
    for i in 2:files
        o = load(string(fname_base, i, ".jld2"))
        AS_cost_ = cat(AS_cost_,o["AS_cost"]; dims=3)
        BS_cost_ = cat(BS_cost_,o["BS_cost"]; dims=3)
    end
    ts_, AS_cost_, BS_cost_, Ns_, maxIter_
end

ts, AS_iter, BS_iter, Ns, maxIter = test_merge()
@save "simpleDemoCompareCostAll_results.jld2" ts AS_iter BS_iter Ns maxIter
