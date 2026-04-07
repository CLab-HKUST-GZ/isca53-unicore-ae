package UniCore.Convert

import spinal.core._
import spinal.core.sim._
import UniCore.Config
import scala.language.postfixOps


// For Activation: FP6 -> FP8
case class E3M2B3toE4M3B7() extends Component {
  val io = new Bundle {
    val E3M2_in  = in  Bits(5 bits)
    val E4M3_out = out Bits(7 bits)
  }
  noIoPrefix()

  // * Extracting
  val E3 = io.E3M2_in(4 downto 2)
  val M2 = io.E3M2_in(1 downto 0)

  val isSubNorm = ~E3.orR
  val E3M2toE4M3_SubNorm = Bits(7 bits)
  val E3M2toE4M3_Norm    = Bits(7 bits)

  // * Subnormal Numbers
  switch(M2) {
    is(B("00"))  { E3M2toE4M3_SubNorm := B"0000_000" }    // value = 0.0
    is(B("01"))  { E3M2toE4M3_SubNorm := B"0011_000" }    // value = 0.0625
    is(B("10"))  { E3M2toE4M3_SubNorm := B"0100_000" }    // value = 0.125
    is(B("11"))  { E3M2toE4M3_SubNorm := B"0100_100" }    // value = 0.1875
  }

  // * Normal Numbers
  val E4 = E3.asUInt +^ U(4)
  val M3 = M2 ## B"0"
  E3M2toE4M3_Norm := E4 ## M3

  // * Output
  io.E4M3_out := Mux(isSubNorm, E3M2toE4M3_SubNorm, E3M2toE4M3_Norm)

}


object E3M2B3toE4M3B7_Gen extends App {
  Config.setGenSubDir("/UniCore")
  Config.spinal.generateVerilog(E3M2B3toE4M3B7()).printRtl().mergeRTLSource()
}