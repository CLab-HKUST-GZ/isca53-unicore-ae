package UniCore.Compensation

import spinal.core._
import spinal.core.sim._
import UniCore.Config
import scala.language.postfixOps


case class Comp_M3_DS() extends Component {
  val io = new Bundle {
    val X_MantMSB = in  Bits(3 bits)
    val Y_MantMSB = in  Bits(3 bits)
    val Comp      = out Bits(1 bits)
  }
  noIoPrefix()

  val InIdx = io.X_MantMSB ## io.Y_MantMSB

  switch(InIdx) {
    is(B"000_000") { io.Comp := B"0" }
    is(B"000_001") { io.Comp := B"0" }
    is(B"000_010") { io.Comp := B"0" }
    is(B"000_011") { io.Comp := B"0" }
    is(B"000_100") { io.Comp := B"0" }
    is(B"000_101") { io.Comp := B"0" }
    is(B"000_110") { io.Comp := B"0" }
    is(B"000_111") { io.Comp := B"0" }
    is(B"001_000") { io.Comp := B"0" }
    is(B"001_001") { io.Comp := B"0" }
    is(B"001_010") { io.Comp := B"0" }
    is(B"001_011") { io.Comp := B"0" }
    is(B"001_100") { io.Comp := B"0" }
    is(B"001_101") { io.Comp := B"0" }
    is(B"001_110") { io.Comp := B"0" }
    is(B"001_111") { io.Comp := B"0" }
    is(B"010_000") { io.Comp := B"0" }
    is(B"010_001") { io.Comp := B"0" }
    is(B"010_010") { io.Comp := B"0" }
    is(B"010_011") { io.Comp := B"1" }
    is(B"010_100") { io.Comp := B"1" }
    is(B"010_101") { io.Comp := B"1" }
    is(B"010_110") { io.Comp := B"0" }
    is(B"010_111") { io.Comp := B"0" }
    is(B"011_000") { io.Comp := B"0" }
    is(B"011_001") { io.Comp := B"0" }
    is(B"011_010") { io.Comp := B"1" }
    is(B"011_011") { io.Comp := B"1" }
    is(B"011_100") { io.Comp := B"1" }
    is(B"011_101") { io.Comp := B"1" }
    is(B"011_110") { io.Comp := B"0" }
    is(B"011_111") { io.Comp := B"0" }
    is(B"100_000") { io.Comp := B"0" }
    is(B"100_001") { io.Comp := B"0" }
    is(B"100_010") { io.Comp := B"1" }
    is(B"100_011") { io.Comp := B"1" }
    is(B"100_100") { io.Comp := B"1" }
    is(B"100_101") { io.Comp := B"1" }
    is(B"100_110") { io.Comp := B"0" }
    is(B"100_111") { io.Comp := B"0" }
    is(B"101_000") { io.Comp := B"0" }
    is(B"101_001") { io.Comp := B"0" }
    is(B"101_010") { io.Comp := B"1" }
    is(B"101_011") { io.Comp := B"1" }
    is(B"101_100") { io.Comp := B"1" }
    is(B"101_101") { io.Comp := B"0" }
    is(B"101_110") { io.Comp := B"0" }
    is(B"101_111") { io.Comp := B"0" }
    is(B"110_000") { io.Comp := B"0" }
    is(B"110_001") { io.Comp := B"0" }
    is(B"110_010") { io.Comp := B"0" }
    is(B"110_011") { io.Comp := B"0" }
    is(B"110_100") { io.Comp := B"0" }
    is(B"110_101") { io.Comp := B"0" }
    is(B"110_110") { io.Comp := B"0" }
    is(B"110_111") { io.Comp := B"0" }
    is(B"111_000") { io.Comp := B"0" }
    is(B"111_001") { io.Comp := B"0" }
    is(B"111_010") { io.Comp := B"0" }
    is(B"111_011") { io.Comp := B"0" }
    is(B"111_100") { io.Comp := B"0" }
    is(B"111_101") { io.Comp := B"0" }
    is(B"111_110") { io.Comp := B"0" }
    is(B"111_111") { io.Comp := B"0" }
  }

}


object Comp_M3_DS_Gen extends App {
  Config.setGenSubDir("/UniCore")
  Config.spinal.generateVerilog(Comp_M3_DS()).printRtl().mergeRTLSource()
}