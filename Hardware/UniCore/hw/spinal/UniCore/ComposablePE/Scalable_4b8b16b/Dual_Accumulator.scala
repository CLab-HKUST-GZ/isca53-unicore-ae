package UniCore.ComposablePE.Scalable_4b8b16b

import spinal.core._
import spinal.core.sim._
import UniCore.Config
import scala.language.postfixOps
import UniCore.Operators.add_int_rca


case class Dual_Accumulator(Width: Int) extends Component {
  val io = new Bundle {
    val SMag_Sign_4b_L_in = in  Bool()
    val SMag_Sign_4b_R_in = in  Bool()
    val SMag_Sign_8b_in   = in  Bool()
    val SMag_Magn_L_in    = in  Bits(Width bits)
    val SMag_Magn_R_in    = in  Bits(Width bits)
    val SInt_L_in         = in  SInt(Width bits)
    val SInt_R_in         = in  SInt(Width bits)

    val PE_Mode           = in  Bool()              // MARK: 0 for Split, 1 for Join

    val SInt_L_out        = out SInt(Width bits)
    val SInt_R_out        = out SInt(Width bits)
  }
  noIoPrefix()


  // * Sign selecting & extending
  val Sign_L = Mux(io.PE_Mode, io.SMag_Sign_8b_in, io.SMag_Sign_4b_L_in)    // This one is actually useless
  val Sign_R = Mux(io.PE_Mode, io.SMag_Sign_8b_in, io.SMag_Sign_4b_R_in)    // But this one is useful
  val ExtendSign_L = Sign_L #* Width      // Which is the same as {Width{Sign}}
  val ExtendSign_R = Sign_R #* Width      // Which is the same as {Width{Sign}}

  val Flip_Magn_L = io.SMag_Magn_L_in ^ ExtendSign_L
  val Flip_Magn_R = io.SMag_Magn_R_in ^ ExtendSign_R


  // * Dual SMag+SInt Adder
  val Adder_L = new add_int_rca(Width=Width)
  val Adder_R = new add_int_rca(Width=Width)

  Adder_R.io.Operand_1 := Flip_Magn_R
  Adder_R.io.Operand_2 := io.SInt_R_in.asBits
  Adder_R.io.Cin := Sign_R.asBits

  Adder_L.io.Operand_1 := Flip_Magn_L
  Adder_L.io.Operand_2 := io.SInt_L_in.asBits
  Adder_L.io.Cin := Mux(io.PE_Mode, Adder_R.io.Cout, Sign_L.asBits)

  io.SInt_L_out := Adder_L.io.Sum.asSInt
  io.SInt_R_out := Adder_R.io.Sum.asSInt


  // * For debug
  val Packed_Mag  = io.SMag_Magn_L_in ## io.SMag_Magn_R_in
  val Packed_SInt_in  = io.SInt_L_in  ## io.SInt_R_in
  val Packed_SInt_out = io.SInt_L_out ## io.SInt_R_out

}


object Dual_Accumulator_Gen extends App {
  Config.setGenSubDir("/UniCore")
  Config.spinal.generateVerilog(Dual_Accumulator(Width=18)).printRtl().mergeRTLSource()
}