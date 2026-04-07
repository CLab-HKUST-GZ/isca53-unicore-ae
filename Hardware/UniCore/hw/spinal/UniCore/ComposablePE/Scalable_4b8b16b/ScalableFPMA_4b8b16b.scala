package UniCore.ComposablePE.Scalable_4b8b16b

import spinal.core._
import spinal.core.sim._
import UniCore.Config
import scala.language.postfixOps
import UniCore.Operators.add_int_rca
import UniCore.Compensation.{Comp_M3_DS, Comp_M3_US, Comp_M2_US}
import UniCore.Convert.{Bin2FPCvt, FP2BinCvt}


// MARK: Need two stages of FPMA
// 3/4b Mode: [0] [1] [2] [3]
// 6/8b Mode: [0 , 1] [2 , 3]
// 16b  Mode: [0 , 1 , 2 , 3]
case class ScalableFPMA_4b8b16b() extends Component {
  val io = new Bundle {
    val A_5b           = in  Vec(Bits(5 bits), 4)       // 3/4b Mode: E3M2 (B=3)
    val W_5b           = in  Vec(Bits(5 bits), 4)       // 3/4b Mode: E3M2 (B=3)
    val NegB           = in  Vec(Bits(5 bits), 4)       // 3/4b Mode: "10100", 6/8b Mode: "11110_01000", 16b Mode:
    val PE_Mode        = in  Bits(2 bits)               // MARK: "00"/"01" for 3/4bits-PE, "10" for 6/8bits-PE, "11" for 16bits-PE

    val M_E4M4_B3_HR   = out (Vec(Bits(8 bits), 4)).simPublic()
    val M_E5M4_B7_HR   = out (Vec(Bits(9 bits), 2)).simPublic()
    val FP8_UnderFlow  = out (Vec(Bool(), 2)).simPublic()
    val M_E5M10_B15    = out (Bits(15 bits)).simPublic()
    val FP16_UnderFlow = out (Bool()).simPublic()

    val SP_ShouldMod   = out Vec(Bool(), 4)
    val is4bMode       = out Bool()
    val is8bMode       = out Bool()
    val is16bMode      = out Bool()
  }
  noIoPrefix()

  // * Down-Sampling Compensation
  val Comp_M3 = (0 until 2).map{ i =>
    new Comp_M3_DS()
  }

  for (i <- 0 until 2) {
    Comp_M3(i).io.X_MantMSB := io.A_5b(i*2+1)(2 downto 0)
    Comp_M3(i).io.Y_MantMSB := io.W_5b(i*2+1)(2 downto 0)
  }

  val Comp_M10 = new DnsmplComp_M10_Comb()       // Comp Value 8b
  Comp_M10.io.A_idx := io.A_5b(2)(4 downto 2)
  Comp_M10.io.W_idx := io.W_5b(2)(4 downto 2)


  // * Mode
  val is4bMode  = io.PE_Mode === B"00" | io.PE_Mode === B"01"
  val is8bMode  = io.PE_Mode === B"10"
  val is16bMode = io.PE_Mode === B"11"


  // * FPMA Stage 1
  val Adder_S1 = (0 until 4).map{ i =>
    new add_int_rca(Width=5)
  }

  for (i <- 0 until 4) {
    Adder_S1(i).io.Operand_1 := io.A_5b(i)
    Adder_S1(i).io.Operand_2 := io.W_5b(i)
  }

  Adder_S1(0).io.Cin := Mux(is16bMode | is8bMode, Adder_S1(1).io.Cout, B"0")
  Adder_S1(1).io.Cin := Mux(is16bMode           , Adder_S1(2).io.Cout, B"0")
  Adder_S1(2).io.Cin := Mux(is16bMode | is8bMode, Adder_S1(3).io.Cout, B"0")
  Adder_S1(3).io.Cin := B"0"

  val Debug_S1 = Adder_S1(0).io.Sum ## Adder_S1(1).io.Sum ## Adder_S1(2).io.Sum ## Adder_S1(3).io.Sum


  // [0] [1] [2] [3]
  val Correction = Vec(Bits(5 bits), 4)
  Correction(0) := io.NegB(0)
  Correction(1) := io.NegB(1)(4 downto 1) ## Mux(is8bMode, Comp_M3(0).io.Comp.asBool, io.NegB(1)(0))    // Insert M3 DS compensation for (1)
  Correction(2) := Mux(is16bMode, io.NegB(2)(4 downto 3) ## Comp_M10.io.Comp(7 downto 5), io.NegB(2))
  Correction(3) := Mux(is16bMode, Comp_M10.io.Comp(4 downto 0), io.NegB(3)(4 downto 1) ## Mux(is8bMode, Comp_M3(1).io.Comp, B"0"))    // Insert M3 DS compensation for (3)


  // * FPMA Stage 2
  val Adder_S2 = (0 until 4).map{ i =>
    new add_int_rca(Width=5)
  }

  for (i <- 0 until 4) {
    Adder_S2(i).io.Operand_1 := Adder_S1(i).io.Sum
    Adder_S2(i).io.Operand_2 := Correction(i)
  }

  Adder_S2(0).io.Cin := Mux(is16bMode | is8bMode, Adder_S2(1).io.Cout, B"0")
  Adder_S2(1).io.Cin := Mux(is16bMode           , Adder_S2(2).io.Cout, B"0")
  Adder_S2(2).io.Cin := Mux(is16bMode | is8bMode, Adder_S2(3).io.Cout, B"0")
  Adder_S2(3).io.Cin := B"0"

  val Debug_Cr = Correction(0) ## Correction(1) ## Correction(2) ## Correction(3)
  val Debug_S2 = Adder_S2(0).io.Sum ## Adder_S2(1).io.Sum ## Adder_S2(2).io.Sum ## Adder_S2(3).io.Sum


  val ExtraBit = (0 until 4).map{ i =>
    Adder_S1(i).io.Cout & Adder_S2(i).io.Cout
  }


  // * Up-Scaling Compensation
  val Comp_HR_M2 = (0 until 4).map{ i =>
    new Comp_M2_US()
  }

  for (i <- 0 until 4) {
    Comp_HR_M2(i).io.A_M2 := io.A_5b(i)(1 downto 0)
    Comp_HR_M2(i).io.W_M2 := io.W_5b(i)(1 downto 0)
  }

  val Comp_HR_M3 = (0 until 2).map{ i =>
    new Comp_M3_US()
  }

  for (i <- 0 until 2) {
    Comp_HR_M3(i).io.A_M3 := io.A_5b(i*2+1)(2 downto 0)
    Comp_HR_M3(i).io.W_M3 := io.W_5b(i*2+1)(2 downto 0)
  }


  // * Special Compensation (Only for A = 0.5 & W = 0.25)
  for (i <- 0 until 4) {
    io.SP_ShouldMod(i) := (io.A_5b(i) === B"01000" & io.W_5b(i) === B"00100")
  }

  val Comp_SP = (0 until 4).map{ i =>
    Mux(io.SP_ShouldMod(i), True, Adder_S2(i).io.Sum(1))
  }


  // * Results
  for (i <- 0 until 4) {
    io.M_E4M4_B3_HR(i) := ExtraBit(i) ## Adder_S2(i).io.Sum(4 downto 2) ## Comp_SP(i) ## Adder_S2(i).io.Sum(0) ## Comp_HR_M2(i).io.HyperRes
  }

  for (i <- 0 until 2) {
    io.M_E5M4_B7_HR(i) := Adder_S2(i*2).io.Sum(2 downto 0) ## Adder_S2(i*2+1).io.Sum ## Comp_HR_M3(i).io.HyperRes
    io.FP8_UnderFlow(i) := Adder_S2(i*2).io.Sum(4)      // Meaning that A+W+C-B < 0
  }

  io.M_E5M10_B15 := Adder_S2(1).io.Sum ## Adder_S2(2).io.Sum ## Adder_S2(3).io.Sum
  io.FP16_UnderFlow := Adder_S2(0).io.Sum(4)      // Meaning that A+W+C-B < 0

  io.is4bMode  := is4bMode
  io.is8bMode  := is8bMode
  io.is16bMode := is16bMode

}


object ScalableFPMA_4b8b16b_Gen extends App {
  Config.setGenSubDir("/UniCore")
  Config.spinal.generateVerilog(ScalableFPMA_4b8b16b()).printRtl().mergeRTLSource()
}