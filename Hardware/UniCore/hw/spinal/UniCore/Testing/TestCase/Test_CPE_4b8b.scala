package UniCore.Testing.TestCase

import spinal.core._
import spinal.core.sim._
import UniCore.Config
import scala.language.postfixOps
import UniCore.ComposablePE.Scalable_4b8b.{PreAdd_4b8b, ScalablePE_4b8b}
import UniCore.Convert.{Bin2FPCvt, FP2BinCvt}
import UniCore.Testing.GoldenModels.{SFPMA_8b, SFPMA_16b}


case class Test_CPE_Top_4b8b(AccuWidth: Int=18) extends Component {
  val io = new Bundle {
    val A_6b_L           = in  Bits(6 bits)     // 3/4b Mode: S1E3M2 (B=3)
    val A_6b_R           = in  Bits(6 bits)     // 3/4b Mode: S1E3M2 (B=3)
    val W_6b_L           = in  Bits(6 bits)     // 3/4b Mode: S1E3M2 (B=3)
    val W_6b_R           = in  Bits(6 bits)     // 3/4b Mode: S1E3M2 (B=3)

    val PE_Mode          = in  Bool()           // MARK: 0 for W4A4, 1 for W8A8

    val PSumIn_L         = in  SInt(AccuWidth bits)
    val PSumIn_R         = in  SInt(AccuWidth bits)

    val M_S1E4M4_B3_HR_L = out Bits(9  bits)    // W4A4 S-FPMA Product
    val M_S1E4M4_B3_HR_R = out Bits(9  bits)    // W4A4 S-FPMA Product
    val M_S1E5M4_B7_HR   = out Bits(10 bits)    // W8A8 S-FPMA Product
    val M_W8_UnderFlow   = out Bool()           // W8A8 S-FPMA Product UnderFlow

    val PSumOut_L        = out SInt(AccuWidth bits)
    val PSumOut_R        = out SInt(AccuWidth bits)
  }
  noIoPrefix()

  // * Pre-Add
  val PreAdd = new PreAdd_4b8b()
  PreAdd.io.A_6b_L := io.A_6b_L
  PreAdd.io.A_6b_R := io.A_6b_R
  PreAdd.io.PE_Mode := io.PE_Mode

  // * Composable PE (Basic)
  val C_PE = new ScalablePE_4b8b(DebugMode=true)
  C_PE.io.T_7b_CIN_L := PreAdd.io.T_7b_L
  C_PE.io.T_7b_CIN_R := PreAdd.io.T_7b_R
  C_PE.io.W_6b_L := io.W_6b_L
  C_PE.io.W_6b_R := io.W_6b_R
  C_PE.io.PE_Mode := io.PE_Mode
  C_PE.io.PSumIn_L := io.PSumIn_L
  C_PE.io.PSumIn_R := io.PSumIn_R

  // * Debug
  io.M_S1E4M4_B3_HR_L := C_PE.io.M_S1E4M4_B3_HR_L.get
  io.M_S1E4M4_B3_HR_R := C_PE.io.M_S1E4M4_B3_HR_R.get
  io.M_S1E5M4_B7_HR := C_PE.io.M_S1E5M4_B7_HR.get
  io.M_W8_UnderFlow := C_PE.io.M_W8_UnderFlow.get

  // * Outputs
  io.PSumOut_L := C_PE.io.PSumOut_L
  io.PSumOut_R := C_PE.io.PSumOut_R

}


object Test_CPE_Top_4b8b_Gen extends App {
  Config.setGenSubDir("/UniCore")
  Config.spinal.generateVerilog(Test_CPE_Top_4b8b()).printRtl().mergeRTLSource()
}



object Test_CPE_4b8b {

  def main(args: Array[String]): Unit = {
    runW4A4Mode()
    runW8A8Mode()
  }

  def runAll(): Unit = {
    runW4A4Mode()
    runW8A8Mode()
  }

  def runW4A4Mode(): Unit = {
    // ==========================================
    // W4A4 Mode
    // ==========================================
    println(s"\n${"="*72}")
    println(s"[Test] Composable PE (4b-8b) test in W4A4 Mode:")
    println(s"${"-"*72}")
    println(s"[*] NOTE:")
    println(s"    - Target      : PE-level MAC (Multiply-Accumulate) output.")
    println(s"    - Equation    : PSumOut = A * W + PSumIn.")
    println(s"    - Fmt support : Covering range for A in FP4, W in DynFP3/4.")
    println(s"    - Parallelism : 2 split channel.")
    println(s"    - PSum Format : Fixed-point representation with 4 fractional bits.")
    println(s"${"="*72}\n")
    runTest(PE_Mode=false)    // MARK: FP3/4
  }

  def runW8A8Mode(): Unit = {
    // ==========================================
    // W8A8 Mode
    // ==========================================
    println(s"\n${"="*80}")
    println(s"[Test] Composable PE (4b-8b) test in W8A8 Mode:")
    println(s"${"-"*80}")
    println(s"[*] NOTE:")
    println(s"    - Target      : PE-level MAC output.")
    println(s"    - Equation    : PSumOut = A * W + PSumIn.")
    println(s"    - Fmt support : Covering range for A in FP6/8, W in FP6/8.")
    println(s"    - Parallelism : 1 combined channel.")
    println(s"    - PSum Format : Fixed-point representation with 11 fractional bits.")
    println(s"    - UnderFlow   : A [UF] tag triggers the Flush-to-Zero (FTZ) protect.")
    println(s"${"="*80}\n")
    runTest(PE_Mode=true)    // MARK: FP6/8
  }


  def runTest(PE_Mode: Boolean): Unit = {

    // Test params
    var PSumIn_W4A4_L = 100
    var PSumIn_W4A4_R = 200
    var PSumIn_W8A8   = 100    // Value can be changed

    // W8A8 PSumIn Preprocessing
    val PSumIn_W8A8_36b = PSumIn_W8A8.toLong << 11
    val Lower_18b = PSumIn_W8A8_36b & 0x3FFFF
    val Upper_18b = (PSumIn_W8A8_36b >> 18) & 0x3FFFF
    val Lower_SInt = if (Lower_18b >= 131072) Lower_18b - 262144 else Lower_18b
    val Upper_SInt = if (Upper_18b >= 131072) Upper_18b - 262144 else Upper_18b

    // Temp variables
    var A_Idx = 0
    var W_Idx = 0


    // * Hardware Simulation
    Config.sim.compile{Test_CPE_Top_4b8b()}.doSim { dut =>
      dut.clockDomain.forkStimulus(2)
      if (!PE_Mode) {    // MARK: W4A4
        for (clk <- 0 until 400) {
          // test case
          if (clk >= 10 && clk < (TU.A4_ValueSpace.length * TU.W4_ValueSpace.length)+10) {
            // FP3 & FP4
            A_Idx = (clk - 10) / TU.W4_ValueSpace.length
            W_Idx = (clk - 10) % TU.W4_ValueSpace.length
            dut.io.A_6b_L   #= TU.A4_ValueSpace(A_Idx)
            dut.io.A_6b_R   #= TU.A4_ValueSpace(A_Idx)
            dut.io.W_6b_L   #= TU.W4_ValueSpace(W_Idx)
            dut.io.W_6b_R   #= TU.W4_ValueSpace(W_Idx)
            dut.io.PE_Mode  #= PE_Mode
            dut.io.PSumIn_L #= PSumIn_W4A4_L << 4
            dut.io.PSumIn_R #= PSumIn_W4A4_R << 4
          } else {
            dut.io.A_6b_L   #= 0
            dut.io.A_6b_R   #= 0
            dut.io.W_6b_L   #= 0
            dut.io.W_6b_R   #= 0
            dut.io.PE_Mode  #= PE_Mode
            dut.io.PSumIn_L #= 0
            dut.io.PSumIn_R #= 0
          }
          dut.clockDomain.waitRisingEdge()    // sample on rising edge

          // * Print result
          // FP3 & FP4
          if (clk >= 10 && clk < (TU.A4_ValueSpace.length * TU.W4_ValueSpace.length)+10) {
            // === A ===
            val A_6b_L_FPBin = String.format(s"%6s", dut.io.A_6b_L.toInt.toBinaryString).replace(' ', '0')    // E3M2_B3
            val A_6b_R_FPBin = String.format(s"%6s", dut.io.A_6b_R.toInt.toBinaryString).replace(' ', '0')    // E3M2_B3
            val A_6b_L_FP = Bin2FPCvt.FPAnyBinToFloat(FPBin=A_6b_L_FPBin, ExpoWidth=3, MantWidth=2, CustomBias=Some(3), WithNaNInf=false)
            val A_6b_R_FP = Bin2FPCvt.FPAnyBinToFloat(FPBin=A_6b_R_FPBin, ExpoWidth=3, MantWidth=2, CustomBias=Some(3), WithNaNInf=false)
            // === W ===
            val W_6b_L_FPBin = String.format(s"%6s", dut.io.W_6b_L.toInt.toBinaryString).replace(' ', '0')    // E3M2_B3
            val W_6b_R_FPBin = String.format(s"%6s", dut.io.W_6b_R.toInt.toBinaryString).replace(' ', '0')    // E3M2_B3
            val W_6b_L_FP = Bin2FPCvt.FPAnyBinToFloat(FPBin='0'+W_6b_L_FPBin, ExpoWidth=4, MantWidth=2, CustomBias=Some(3), WithNaNInf=false)
            val W_6b_R_FP = Bin2FPCvt.FPAnyBinToFloat(FPBin='0'+W_6b_R_FPBin, ExpoWidth=4, MantWidth=2, CustomBias=Some(3), WithNaNInf=false)
            // === M ===
            val M_L_FPBin = String.format(s"%9s", dut.io.M_S1E4M4_B3_HR_L.toInt.toBinaryString).replace(' ', '0')
            val M_R_FPBin = String.format(s"%9s", dut.io.M_S1E4M4_B3_HR_R.toInt.toBinaryString).replace(' ', '0')
            val M_L_FP = Bin2FPCvt.FPAnyBinToFloat(FPBin=M_L_FPBin, ExpoWidth=4, MantWidth=4, CustomBias=Some(3), WithNaNInf=false)
            val M_R_FP = Bin2FPCvt.FPAnyBinToFloat(FPBin=M_R_FPBin, ExpoWidth=4, MantWidth=4, CustomBias=Some(3), WithNaNInf=false)
            // === PSum ===
            val PSum_L = dut.io.PSumOut_L.toInt.toFloat / 16
            val PSum_R = dut.io.PSumOut_R.toInt.toFloat / 16
            // === Golden Model Execution ===
            val RealProd_L = A_6b_L_FP * W_6b_L_FP
            val RealProd_R = A_6b_R_FP * W_6b_R_FP
            val GoldenPSum_L = RealProd_L + PSumIn_W4A4_L
            val GoldenPSum_R = RealProd_R + PSumIn_W4A4_R
            // === Compare & Print ===
            val Compare = TU.checkResult(M_L_FP == RealProd_L && M_R_FP == RealProd_R && PSum_L == GoldenPSum_L && PSum_R == GoldenPSum_R)
            val A_str_L   = s"A: %-6s".format(A_6b_L_FP)
            val W_str_L   = s"W: %-6s".format(W_6b_L_FP)
            val PIn_str_L = s"PIn: %-4s".format(PSumIn_W4A4_L.toFloat)
            val DUT_L     = s"[DUT] PSum=%-8s".format(PSum_L)
            val GLD_L     = s"[GLD Ax] PSum=%-8s".format(GoldenPSum_L)
            val A_str_R   = s"A: %-6s".format(A_6b_R_FP)
            val W_str_R   = s"W: %-6s".format(W_6b_R_FP)
            val PIn_str_R = s"PIn: %-4s".format(PSumIn_W4A4_R.toFloat)
            val DUT_R     = s"[DUT] PSum=%-8s".format(PSum_R)
            val GLD_R     = s"[GLD Ax] PSum=%-8s".format(GoldenPSum_R)
            val Block_L = s"[Ch0] $A_str_L * $W_str_L + $PIn_str_L  =>  $DUT_L vs $GLD_L"
            val Block_R = s"[Ch1] $A_str_R * $W_str_R + $PIn_str_R  =>  $DUT_R vs $GLD_R"
            printf(s"%-86s | %-84s | %s\n", Block_L, Block_R, s"[Check] $Compare")
          }
        }
      } else  {    // MARK: W8A8
        for (clk <- 0 until 200) {
          // test case
          if (clk >= 10 && clk < 128+10) {
            // FP6 & FP8
            dut.io.A_6b_L   #= (clk - 10) / 64
            dut.io.A_6b_R   #= (clk - 10) % 64
            dut.io.W_6b_L   #= (clk - 10) / 64
            dut.io.W_6b_R   #= (clk - 10) % 64
            dut.io.PE_Mode  #= PE_Mode
            dut.io.PSumIn_L #= Upper_SInt
            dut.io.PSumIn_R #= Lower_SInt
          } else {
            dut.io.A_6b_L   #= 0
            dut.io.A_6b_R   #= 0
            dut.io.W_6b_L   #= 0
            dut.io.W_6b_R   #= 0
            dut.io.PE_Mode  #= PE_Mode
            dut.io.PSumIn_L #= 0
            dut.io.PSumIn_R #= 0
          }
          dut.clockDomain.waitRisingEdge()    // sample on rising edge

          if (clk >= 10 && clk < 128+10) {
            // * Print result
            // FP6 & FP8
            // === A ===
            val A_6b_L_Bin = String.format(s"%6s", dut.io.A_6b_L.toInt.toBinaryString).replace(' ', '0')
            val A_6b_R_Bin = String.format(s"%6s", dut.io.A_6b_R.toInt.toBinaryString).replace(' ', '0')
            val A_E4M3_Bin = A_6b_L_Bin.substring(5, 6) + A_6b_R_Bin
            val A_E4M3_FP  = Bin2FPCvt.FPAnyBinToFloat(FPBin='0'+A_E4M3_Bin, ExpoWidth=4, MantWidth=3, CustomBias=Some(7), WithNaNInf=false)
            // === W ===
            val W_6b_L_Bin = String.format(s"%6s", dut.io.W_6b_L.toInt.toBinaryString).replace(' ', '0')
            val W_6b_R_Bin = String.format(s"%6s", dut.io.W_6b_R.toInt.toBinaryString).replace(' ', '0')
            val W_E4M3_Bin = W_6b_L_Bin.substring(5, 6) + W_6b_R_Bin
            val W_E4M3_FP  = Bin2FPCvt.FPAnyBinToFloat(FPBin='0'+W_E4M3_Bin, ExpoWidth=4, MantWidth=3, CustomBias=Some(7), WithNaNInf=false)
            // === Golden Model Execution ===
            val A_8b_BigInt = BigInt("0" + A_E4M3_Bin, 2)
            val W_8b_BigInt = BigInt("0" + W_E4M3_Bin, 2)
            // Golden Model Result (Multiplication)
            val Golden_M_Raw_Bin = SFPMA_8b.calculate(W_8b_BigInt, A_8b_BigInt)
            val Golden_M_Raw_FP  = Bin2FPCvt.FPAnyBinToFloat(FPBin=Golden_M_Raw_Bin, ExpoWidth=5, MantWidth=4, CustomBias=Some(7), WithNaNInf=false, WithSubnorm=false)
            val isUnderFlow = dut.io.M_W8_UnderFlow.toBoolean
            val Golden_M_FP = if (isUnderFlow) 0.0 else Golden_M_Raw_FP    // Flush-To-Zero if UnderFlow
            // [DUT] PSum Execution
            val PSum_L = BigInt(dut.io.PSumOut_L.toInt)
            val PSum_R = dut.io.PSumOut_R.toBigInt & ((BigInt(1) << 18) - 1)
            val PSum = (PSum_L << 7).toDouble + (PSum_R.toDouble / 2048)
            // === Golden PSum & Compare ===
            val GoldenPSum = Golden_M_FP + PSumIn_W8A8
            val Compare = TU.checkResult(PSum == GoldenPSum)
            // === Binary Conversion for PSum Compare ===
            val dut_l_bin = String.format("%18s", (dut.io.PSumOut_L.toInt & 0x3FFFF).toBinaryString).replace(' ', '0')
            val dut_r_bin = String.format("%18s", (dut.io.PSumOut_R.toInt & 0x3FFFF).toBinaryString).replace(' ', '0')
            val DUT_Bin   = s"${dut_l_bin}_${dut_r_bin}"
            // === Format & Print ===
            val A_str   = s"A: %-11s".format(A_E4M3_FP)
            val W_str   = s"W: %-11s".format(W_E4M3_FP)
            val PIn_str = s"PIn: %-4s".format(PSumIn_W8A8.toDouble)
            val DUT_str = s"[DUT] PSum = %-16s".format(PSum)
            val GLD_str = s"[Golden Approx] PSum=%-16s".format(GoldenPSum)
            val Block = s"[W8A8] $A_str * $W_str + $PIn_str  =>  $DUT_str vs $GLD_str"
            // === Exact Math Execution ===
            val ExactPSum = (A_E4M3_FP * W_E4M3_FP) + PSumIn_W8A8
            val Exact_str = s"[Exact Math] PSum=%-18s".format(ExactPSum)
            val UF_Tag = if (isUnderFlow) s"\u001B[35m[UF]\u001B[0m " else "     "
            printf(s"%-125s | %s%-26s | %s\n", Block, UF_Tag, s"[Check] $Compare", Exact_str)
          }
        }
      }
      sleep(50)
    }

  }


}
