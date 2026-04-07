package UniCore.ComposablePE.Scalable_4b8b

import spinal.core._
import spinal.core.sim._
import UniCore.Config
import scala.language.postfixOps
import UniCore.Compensation.Comp_M2_US
import UniCore.Operators.add_int_rca


// MARK: Uniform-Precision (W4A4 & W8A8)
case class ScalableFPMA_4b8b() extends Component {
  val io = new Bundle {
    val T_7b_L           = in  Bits(7 bits)     // Unified 7b T for L
    val T_7b_R           = in  Bits(7 bits)     // Unified 7b T for R
    val W_6b_L           = in  Bits(6 bits)     // 3/4b Mode: S1E3M2 (B=3)
    val W_6b_R           = in  Bits(6 bits)     // 3/4b Mode: S1E3M2 (B=3)

    val PE_Mode          = in  Bool()           // MARK: 0 for W4A4, 1 for W8A8

    // Compensation for W8A8
    val Comp_M3_DS_Val   = in  Bits(1 bits)     // Down-Sampling Compensation for W8A8
    val Comp_M3_US_Val   = in  Bits(1 bits)     //    Up-Scaling Compensation for W8A8

    val M_S1E4M4_B3_HR_L = out Bits(9  bits)    // W4A4
    val M_S1E4M4_B3_HR_R = out Bits(9  bits)    // W4A4
    val M_S1E5M4_B7_HR   = out Bits(10 bits)    // W8A8
    val FP8_UnderFlow    = out Bool()
  }
  noIoPrefix()

  // * Extracting
  val Sign_A4_L = io.T_7b_L(6)    // Sign for A4_L or A8
  val Sign_A4_R = io.T_7b_R(6)    // Sign for A4_R

  val Sign_W4_L = io.W_6b_L(5)    // Sign for W4_L or W8
  val Sign_W4_R = io.W_6b_R(5)    // Sign for W4_R

  // * Preparing
  val Op_W_6b_L = B"0" ## io.W_6b_L(4 downto 0)
  val Op_W_6b_R = Mux(io.PE_Mode, io.W_6b_R(5).asBits, B"0") ## io.W_6b_R(4 downto 0)


  // * FPMA Stage 2
  val Adder_S2_L = new add_int_rca(Width=6)
  val Adder_S2_R = new add_int_rca(Width=6)

  Adder_S2_R.io.Operand_1 := io.T_7b_R(5 downto 0)
  Adder_S2_R.io.Operand_2 := Op_W_6b_R
  Adder_S2_R.io.Cin := Mux(io.PE_Mode, io.Comp_M3_DS_Val, B"0")

  Adder_S2_L.io.Operand_1 := io.T_7b_L(5 downto 0)
  Adder_S2_L.io.Operand_2 := Op_W_6b_L
  Adder_S2_L.io.Cin := Mux(io.PE_Mode, Adder_S2_R.io.Cout, B"0")

  // * Sign
  val Sign_M4_L = Sign_A4_L ^ Sign_W4_L
  val Sign_M4_R = Sign_A4_R ^ Sign_W4_R
  val Sign_M8   = Sign_M4_L

  // * Up-Scaling Compensation
  // W4A4
  val Comp_M2_US_L = new Comp_M2_US()
  val Comp_M2_US_R = new Comp_M2_US()
  Comp_M2_US_L.io.A_M2 := io.T_7b_L(1 downto 0)
  Comp_M2_US_L.io.W_M2 := io.W_6b_L(1 downto 0)
  Comp_M2_US_R.io.A_M2 := io.T_7b_R(1 downto 0)
  Comp_M2_US_R.io.W_M2 := io.W_6b_R(1 downto 0)

  // * Outputs
  // W4A4
  io.M_S1E4M4_B3_HR_L := Sign_M4_L ## Adder_S2_L.io.Sum ## Comp_M2_US_L.io.HyperRes
  io.M_S1E4M4_B3_HR_R := Sign_M4_R ## Adder_S2_R.io.Sum ## Comp_M2_US_R.io.HyperRes
  // W8A8
  io.M_S1E5M4_B7_HR := Sign_M8 ## Adder_S2_L.io.Sum(1 downto 0) ## Adder_S2_R.io.Sum ## io.Comp_M3_US_Val
  io.FP8_UnderFlow := Adder_S2_L.io.Sum(5)      // Meaning that A+W+C-B < 0

}


object ScalableFPMA_4b8b_Gen extends App {
  Config.setGenSubDir("/UniCore")
  Config.spinal.generateVerilog(ScalableFPMA_4b8b()).printRtl().mergeRTLSource()
}