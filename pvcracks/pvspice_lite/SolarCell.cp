.subckt Cell a b c PARAMS: Rs=10 Rp=2k I=1 BV=30 N=1.5 Is=1
    .model sd1 D (IS={Is} N={N} EG=1.12 BV={BV})
    Rs1 a b {Rs}
    Rp1 b c {Rp}
    D1 b c sd1
    I1 c b {I}
    .temp = 25
    .options tnom=25
.ends