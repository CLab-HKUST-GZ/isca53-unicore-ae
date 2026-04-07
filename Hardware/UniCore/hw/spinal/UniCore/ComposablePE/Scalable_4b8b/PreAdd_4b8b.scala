package UniCore.ComposablePE.Scalable_4b8b

import spinal.core._
import spinal.core.sim._
import UniCore.Config
import scala.language.postfixOps
import UniCore.Operators.add_int_rca


// Pre Adder, which is the first stage of FPMA taken out
case class PreAdd_4b8b() extends Component {
  val io = new Bundle {
    val A_6b_L  = in  Bits(6 bits)    // 3/4b Mode: S1E3M2 (B=3)
    val A_6b_R  = in  Bits(6 bits)    // 3/4b Mode: S1E3M2 (B=3)

    val PE_Mode = in  Bool()          // MARK: 0 for W4A4, 1 for W8A8

    val T_7b_L  = out Bits(7 bits)    // Unified 7b T for L
    val T_7b_R  = out Bits(7 bits)    // Unified 7b T for R
  }
  noIoPrefix()

  // * Extracting
  val Sign_A4_L = io.A_6b_L(5)    // Sign for A4_L or A8
  val Sign_A4_R = io.A_6b_R(5)    // Sign for A4_R
  val Op_A_6b_L = B"0" ## io.A_6b_L(4 downto 0)
  val Op_A_6b_R = Mux(io.PE_Mode, io.A_6b_R(5).asBits, B"0") ## io.A_6b_R(4 downto 0)


  // * -B
  val NegB_W4A4_E3M1B3 = B"1101" ## B"00"                 // -3 => 1101, Total 6 bits
  val NegB_W8A8_E4M3B7 = B"11111" ## B"1001" ## B"000"    // -7 => 1001, Total 12 bits


  // * FPMA Stage 1
  val Adder_S1_L = new add_int_rca(Width=6)
  val Adder_S1_R = new add_int_rca(Width=6)

  Adder_S1_R.io.Operand_1 := Op_A_6b_R
  Adder_S1_R.io.Operand_2 := Mux(io.PE_Mode, NegB_W8A8_E4M3B7(5 downto 0), NegB_W4A4_E3M1B3)
  Adder_S1_R.io.Cin := B"0"

  Adder_S1_L.io.Operand_1 := Op_A_6b_L
  Adder_S1_L.io.Operand_2 := Mux(io.PE_Mode, NegB_W8A8_E4M3B7(11 downto 6), NegB_W4A4_E3M1B3)
  Adder_S1_L.io.Cin := Mux(io.PE_Mode, Adder_S1_R.io.Cout, B"0")


  // * Results
  val T_6b_L = Adder_S1_L.io.Sum    // 6 bits
  val T_6b_R = Adder_S1_R.io.Sum    // 6 bits

  io.T_7b_L := Sign_A4_L ## T_6b_L
  io.T_7b_R := Sign_A4_R ## T_6b_R

}


object PreAdd_4b8b_Gen extends App {
  Config.setGenSubDir("/UniCore")
  Config.spinal.generateVerilog(PreAdd_4b8b()).printRtl().mergeRTLSource()
}