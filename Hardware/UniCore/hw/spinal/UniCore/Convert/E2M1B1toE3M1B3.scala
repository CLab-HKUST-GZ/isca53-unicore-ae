package UniCore.Convert

import spinal.core._
import spinal.core.sim._
import UniCore.Config
import scala.language.postfixOps


// For Activation: FP4 -> E3M1
// This will reduce all subnormal numbers
case class E2M1B1toE3M1B3() extends Component {
  val io = new Bundle {
    val E2M1_in  = in  Bits(3 bits)
    val E3M1_out = out Bits(4 bits)
  }
  noIoPrefix()

  switch(io.E2M1_in) {
    is(B("00_0")) { io.E3M1_out := B"000_0" }    // 0
    is(B("00_1")) { io.E3M1_out := B"010_0" }    // 0.5
    is(B("01_0")) { io.E3M1_out := B"011_0" }    // 1
    is(B("01_1")) { io.E3M1_out := B"011_1" }    // 1.5
    is(B("10_0")) { io.E3M1_out := B"100_0" }    // 2
    is(B("10_1")) { io.E3M1_out := B"100_1" }    // 3
    is(B("11_0")) { io.E3M1_out := B"101_0" }    // 4
    is(B("11_1")) { io.E3M1_out := B"101_1" }    // 6
  }
}


object E2M1B1toE3M1B3_Gen extends App {
  Config.setGenSubDir("/UniCore")
  Config.spinal.generateVerilog(E2M1B1toE3M1B3()).printRtl().mergeRTLSource()
}