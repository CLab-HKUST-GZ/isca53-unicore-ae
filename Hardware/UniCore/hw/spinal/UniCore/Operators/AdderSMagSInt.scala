package UniCore.Operators

import spinal.core._
import spinal.core.sim._
import UniCore.Config
import scala.language.postfixOps
import UniCore.Operators.add_int_rca


case class AdderSMagSInt(Width: Int) extends Component {
  val io = new Bundle {
    val SMag_Sign_in = in  Bits(1 bits)
    val SMag_Magn_in = in  Bits(Width bits)
    val SInt_in      = in  SInt(Width bits)
    val SInt_out     = out SInt(Width bits)
  }
  noIoPrefix()

  val ExtendSign = io.SMag_Sign_in #* Width      // Which is the same as {Width{Sign}}
  val Flip_Magn = io.SMag_Magn_in ^ ExtendSign
  val Adder = new add_int_rca(Width=Width)
  Adder.io.Cin := io.SMag_Sign_in.asBits
  Adder.io.Operand_1 := Flip_Magn
  Adder.io.Operand_2 := io.SInt_in.asBits

  io.SInt_out := Adder.io.Sum.asSInt
}


object AdderSMagSInt_Gen extends App {
  Config.setGenSubDir("/UniCore")
  Config.spinal.generateVerilog(AdderSMagSInt(Width=18)).printRtl().mergeRTLSource()
}



object AdderSMagSInt_Sim extends App {
  Config.sim.compile{AdderSMagSInt(Width=14)}.doSim { dut =>
    // simulation process
    dut.clockDomain.forkStimulus(2)
    // simulation code
    for (clk <- 0 until 100) {
      // test case
      if (clk >= 10 && clk < 80+10) {
        dut.io.SMag_Sign_in #= 1
        dut.io.SMag_Magn_in #= 96
        dut.io.SInt_in      #= -(96*63)
      } else {
        dut.io.SMag_Sign_in #= 0
        dut.io.SMag_Magn_in #= 0
        dut.io.SInt_in      #= 0
      }
      dut.clockDomain.waitRisingEdge()    // sample on rising edge
    }
    sleep(50)
  }
}