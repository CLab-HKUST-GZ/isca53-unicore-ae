package UniCore.Convert

import spinal.core._
import spinal.core.sim._
import UniCore.Config
import scala.language.postfixOps


// Convert: DynFP3 (Padded zero below LSB), DynFP4 => S1E3M2
case class DynFP3FP4toS1E3M2() extends Component {
  val io = new Bundle {
    val W_DynFP3FP4     = in  Bits(4 bits)    // DynFP3 or DynFP4
    val DynFP3FP4FmtSel = in  Bits(3 bits)    // [DynFP3]: E2M0I, E2M0, E1M1I, E1M1; [DynFP4]: E1M2I, E1M2, E2M1, E3M0
    val Z_E3M2          = in  Bits(5 bits)
    val Unified_S1E3M2  = out Bits(6 bits)
  }
  noIoPrefix()

  val W_DynFP3FP4_S  = io.W_DynFP3FP4(3)             // (DynFP3 will padded zero below LSB)
  val W_DynFP3FP4_EM = io.W_DynFP3FP4(2 downto 0)    // (DynFP3 will padded zero below LSB)

  val Decide = io.DynFP3FP4FmtSel ## W_DynFP3FP4_EM
  val ConvertedE3M2 = Bits(5 bits)

  // Check whether is Negative 0 in DynFP3/4
  val isNegZero = io.W_DynFP3FP4 === B"1000"

  switch(Decide) {
    // * FP3 (E2M0I) (FmtSel=000)
    is(B("000__00__0"))  { ConvertedE3M2 := B"000_00" }    // value = 0.0    (E0E: 000_0)
    is(B("000__01__0"))  { ConvertedE3M2 := B"011_00" }    // value = 1.0    (E0E: 001_0)
    is(B("000__10__0"))  { ConvertedE3M2 := B"110_00" }    // value = 8.0    (E0E: 100_0)
    is(B("000__11__0"))  { ConvertedE3M2 := B"111_00" }    // value = 16.0   (E0E: 101_0)

    // * FP3 (E2M0) (FmtSel=001)
    is(B("001__00__0"))  { ConvertedE3M2 := B"000_00" }    // value = 0.0
    is(B("001__01__0"))  { ConvertedE3M2 := B"011_00" }    // value = 1.0
    is(B("001__10__0"))  { ConvertedE3M2 := B"100_00" }    // value = 2.0
    is(B("001__11__0"))  { ConvertedE3M2 := B"101_00" }    // value = 4.0

    // * FP3 (E1M1I) (FmtSel=010)
    is(B("010__0_0__0")) { ConvertedE3M2 := B"000_00" }    // value = 0.0    (E0M: 000_0)
    is(B("010__0_1__0")) { ConvertedE3M2 := B"011_00" }    // value = 1.0    (E0M: 000_1)  [Subnorm]
    is(B("010__1_0__0")) { ConvertedE3M2 := B"101_00" }    // value = 4.0    (E0M: 010_0)
    is(B("010__1_1__0")) { ConvertedE3M2 := B"101_10" }    // value = 6.0    (E0M: 010_1)

    // * FP3 (E1M1) (FmtSel=011)
    is(B("011__0_0__0")) { ConvertedE3M2 := B"000_00" }    // value = 0.0
    is(B("011__0_1__0")) { ConvertedE3M2 := B"011_00" }    // value = 1.0  [Subnorm]
    is(B("011__1_0__0")) { ConvertedE3M2 := B"100_00" }    // value = 2.0
    is(B("011__1_1__0")) { ConvertedE3M2 := B"100_10" }    // value = 3.0

    // * FP4 (E1M2I) (FmtSel=100)
    is(B("100__0_00")) { ConvertedE3M2 := B"000_00" }    // value = 0.0    (E0MM: 00_00)
    is(B("100__0_01")) { ConvertedE3M2 := B"010_00" }    // value = 0.5    (E0MM: 00_01)  [Subnorm]
    is(B("100__0_10")) { ConvertedE3M2 := B"011_00" }    // value = 1.0    (E0MM: 00_10)  [Subnorm]
    is(B("100__0_11")) { ConvertedE3M2 := B"011_10" }    // value = 1.5    (E0MM: 00_11)  [Subnorm]
    is(B("100__1_00")) { ConvertedE3M2 := B"101_00" }    // value = 4.0    (E0MM: 10_00)
    is(B("100__1_01")) { ConvertedE3M2 := B"101_01" }    // value = 5.0    (E0MM: 10_01)
    is(B("100__1_10")) { ConvertedE3M2 := B"101_10" }    // value = 6.0    (E0MM: 10_10)
    is(B("100__1_11")) { ConvertedE3M2 := B"101_11" }    // value = 7.0    (E0MM: 10_11)

    // * FP4 (E1M2) (FmtSel=101)
    is(B("101__0_00")) { ConvertedE3M2 := B"000_00" }    // value = 0.0
    is(B("101__0_01")) { ConvertedE3M2 := B"010_00" }    // value = 0.5  [Subnorm]
    is(B("101__0_10")) { ConvertedE3M2 := B"011_00" }    // value = 1.0  [Subnorm]
    is(B("101__0_11")) { ConvertedE3M2 := B"011_10" }    // value = 1.5  [Subnorm]
    is(B("101__1_00")) { ConvertedE3M2 := B"100_00" }    // value = 2.0
    is(B("101__1_01")) { ConvertedE3M2 := B"100_01" }    // value = 2.5
    is(B("101__1_10")) { ConvertedE3M2 := B"100_10" }    // value = 3.0
    is(B("101__1_11")) { ConvertedE3M2 := B"100_11" }    // value = 3.5

    // * FP4 (E2M1) (FmtSel=110)
    is(B("110__00_0")) { ConvertedE3M2 := B"000_00" }    // value = 0.0
    is(B("110__00_1")) { ConvertedE3M2 := B"010_00" }    // value = 0.5  [Subnorm]
    is(B("110__01_0")) { ConvertedE3M2 := B"011_00" }    // value = 1.0
    is(B("110__01_1")) { ConvertedE3M2 := B"011_10" }    // value = 1.5
    is(B("110__10_0")) { ConvertedE3M2 := B"100_00" }    // value = 2.0
    is(B("110__10_1")) { ConvertedE3M2 := B"100_10" }    // value = 3.0
    is(B("110__11_0")) { ConvertedE3M2 := B"101_00" }    // value = 4.0
    is(B("110__11_1")) { ConvertedE3M2 := B"101_10" }    // value = 6.0

    // * FP4 (E3M0) (FmtSel=111)
    is(B("111__000")) { ConvertedE3M2 := B"000_00" }    // value = 0.0
    is(B("111__001")) { ConvertedE3M2 := B"001_00" }    // value = 0.25
    is(B("111__010")) { ConvertedE3M2 := B"010_00" }    // value = 0.5
    is(B("111__011")) { ConvertedE3M2 := B"011_00" }    // value = 1.0
    is(B("111__100")) { ConvertedE3M2 := B"100_00" }    // value = 2.0
    is(B("111__101")) { ConvertedE3M2 := B"101_00" }    // value = 4.0
    is(B("111__110")) { ConvertedE3M2 := B"110_00" }    // value = 8.0
    is(B("111__111")) { ConvertedE3M2 := B"111_00" }    // value = 16.0

    default           { ConvertedE3M2 := B"000_00" }
  }

  // * Muxing output, to replace negative zero with special value Z
  io.Unified_S1E3M2 := Mux(isNegZero, B"0" ## io.Z_E3M2, W_DynFP3FP4_S ## ConvertedE3M2)

}


object DynFP3FP4toS1E3M2_Gen extends App {
  Config.setGenSubDir("/UniCore")
  Config.spinal.generateVerilog(DynFP3FP4toS1E3M2()).printRtl().mergeRTLSource()
}