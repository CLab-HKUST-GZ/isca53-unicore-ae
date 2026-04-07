package UniCore.Testing.TestCase

import spinal.core._
import spinal.core.sim._
import UniCore.Config
import scala.language.postfixOps
import UniCore.ComposablePE.Scalable_4b8b.{PreAdd_4b8b, ScalablePE_4b8b}
import UniCore.Convert.{Bin2FPCvt, FP2BinCvt}
import UniCore.Testing.GoldenModels.SFPMA_8b


case class Test_Top_SFPMA_4b8b() extends Component {
  val io = new Bundle {
    val A_6b_L           = in  Bits(6 bits)     // 3/4b Mode: S1E3M2 (B=3)
    val A_6b_R           = in  Bits(6 bits)     // 3/4b Mode: S1E3M2 (B=3)
    val W_6b_L           = in  Bits(6 bits)     // 3/4b Mode: S1E3M2 (B=3)
    val W_6b_R           = in  Bits(6 bits)     // 3/4b Mode: S1E3M2 (B=3)

    val PE_Mode          = in  Bool()           // MARK: 0 for W4A4, 1 for W8A8

    val M_S1E4M4_B3_HR_L = out Bits(9  bits)    // W4A4
    val M_S1E4M4_B3_HR_R = out Bits(9  bits)    // W4A4
    val M_S1E5M4_B7_HR   = out Bits(10 bits)    // W8A8
    val M_W8_UnderFlow   = out Bool()           // W8A8 UnderFlow
  }
  noIoPrefix()

  // * Pre-Add (S-FPMA stage 1)
  val PreAdd = new PreAdd_4b8b()
  PreAdd.io.A_6b_L := io.A_6b_L
  PreAdd.io.A_6b_R := io.A_6b_R
  PreAdd.io.PE_Mode := io.PE_Mode

  // * Scalable PE (S-FPMA stage 2 is inside the S-PE)
  val S_PE = new ScalablePE_4b8b(DebugMode=true)
  S_PE.io.T_7b_CIN_L := PreAdd.io.T_7b_L
  S_PE.io.T_7b_CIN_R := PreAdd.io.T_7b_R
  S_PE.io.W_6b_L := io.W_6b_L
  S_PE.io.W_6b_R := io.W_6b_R
  S_PE.io.PE_Mode := io.PE_Mode
  S_PE.io.PSumIn_L := S(0)
  S_PE.io.PSumIn_R := S(0)

  // * Getting Approximate Products from S-PE
  io.M_S1E4M4_B3_HR_L := S_PE.io.M_S1E4M4_B3_HR_L.get
  io.M_S1E4M4_B3_HR_R := S_PE.io.M_S1E4M4_B3_HR_R.get
  io.M_S1E5M4_B7_HR := S_PE.io.M_S1E5M4_B7_HR.get
  io.M_W8_UnderFlow := S_PE.io.M_W8_UnderFlow.get

}


object Test_Top_SFPMA_4b8b_Gen extends App {
  Config.setGenSubDir("/UniCore")
  Config.spinal.generateVerilog(Test_Top_SFPMA_4b8b()).printRtl().mergeRTLSource()
}



object Test_SFPMA_4b8b {

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
    println(s"\n${"="*80}")
    println(s"[Test] S-FPMA (4b-8b) test in W4A4 Mode:")
    println(s"${"-"*80}")
    println(s"[*] NOTE:")
    println(s"    - Parallelism : 2 independent output channels [DUT Ch0] and [DUT Ch1].")
    println(s"    - Input A Fmt : FP4 (S1E2M1).")
    println(s"    - Input W Fmt : DynFP3 / DynFP4.")
    println(s"${"="*80}\n")
    runTest(PE_Mode=false)    // MARK: FP3/4
  }

  def runW8A8Mode(): Unit = {
    // ==========================================
    // W8A8 Mode
    // ==========================================
    println(s"\n${"="*98}")
    println(s"[Test] S-FPMA (4b-8b) test in W8A8 Mode:")
    println(s"${"-"*98}")
    println(s"[*] NOTE:")
    println(s"    - Parallelism : 1 combined high-precision output channel.")
    println(s"    - Input A Fmt : FP8 (S1E4M3).")
    println(s"    - Input W Fmt : FP6 (S1E3M2) / FP8 (S1E4M3).")
    println(s"    - UnderFlow   : A [UF] tag will trigger the Flush-to-Zero (FTZ) protect logic in PE.")
    println(s"${"="*98}\n")
    runTest(PE_Mode=true)     // MARK: FP6/8
  }


  def runTest(PE_Mode: Boolean): Unit = {

    var A_Idx = 0
    var W_Idx = 0

    Config.sim.compile{Test_Top_SFPMA_4b8b()}.doSim { dut =>
      // simulation process
      dut.clockDomain.forkStimulus(2)
      // simulation code
      if (!PE_Mode) {    // MARK: W4A4
        for (clk <- 0 until 400) {
          // test case
          if (clk >= 10 && clk < (TU.A4_ValueSpace.length * TU.W4_ValueSpace.length)+10) {
            // FP3 & FP4
            A_Idx = (clk - 10) / TU.W4_ValueSpace.length
            W_Idx = (clk - 10) % TU.W4_ValueSpace.length
            dut.io.A_6b_L  #= TU.A4_ValueSpace(A_Idx)
            dut.io.A_6b_R  #= TU.A4_ValueSpace(A_Idx)
            dut.io.W_6b_L  #= TU.W4_ValueSpace(W_Idx)
            dut.io.W_6b_R  #= TU.W4_ValueSpace(W_Idx)
            dut.io.PE_Mode #= PE_Mode
          } else {
            dut.io.A_6b_L  #= 0
            dut.io.A_6b_R  #= 0
            dut.io.W_6b_L  #= 0
            dut.io.W_6b_R  #= 0
            dut.io.PE_Mode #= PE_Mode
          }
          dut.clockDomain.waitRisingEdge()    // sample on rising edge

          // * Print result
          // FP3 & FP4
          if (clk >= 10 && clk < (TU.A4_ValueSpace.length * TU.W4_ValueSpace.length)+10) {
            // A
            val A_6b_L_FPBin = String.format(s"%6s", dut.io.A_6b_L.toInt.toBinaryString).replace(' ', '0')    // E3M2_B3
            val A_6b_R_FPBin = String.format(s"%6s", dut.io.A_6b_R.toInt.toBinaryString).replace(' ', '0')    // E3M2_B3
            val A_6b_L_FP = Bin2FPCvt.FPAnyBinToFloat(FPBin=A_6b_L_FPBin, ExpoWidth=3, MantWidth=2, CustomBias=Some(3), WithNaNInf=false)
            val A_6b_R_FP = Bin2FPCvt.FPAnyBinToFloat(FPBin=A_6b_R_FPBin, ExpoWidth=3, MantWidth=2, CustomBias=Some(3), WithNaNInf=false)
            // W
            val W_6b_L_FPBin = String.format(s"%6s", dut.io.W_6b_L.toInt.toBinaryString).replace(' ', '0')    // E3M2_B3
            val W_6b_R_FPBin = String.format(s"%6s", dut.io.W_6b_R.toInt.toBinaryString).replace(' ', '0')    // E3M2_B3
            val W_6b_L_FP = Bin2FPCvt.FPAnyBinToFloat(FPBin='0'+W_6b_L_FPBin, ExpoWidth=4, MantWidth=2, CustomBias=Some(3), WithNaNInf=false)
            val W_6b_R_FP = Bin2FPCvt.FPAnyBinToFloat(FPBin='0'+W_6b_R_FPBin, ExpoWidth=4, MantWidth=2, CustomBias=Some(3), WithNaNInf=false)
            // M
            val M_L_FPBin = String.format(s"%9s", dut.io.M_S1E4M4_B3_HR_L.toInt.toBinaryString).replace(' ', '0')
            val M_R_FPBin = String.format(s"%9s", dut.io.M_S1E4M4_B3_HR_R.toInt.toBinaryString).replace(' ', '0')
            val M_L_FP = Bin2FPCvt.FPAnyBinToFloat(FPBin=M_L_FPBin, ExpoWidth=4, MantWidth=4, CustomBias=Some(3), WithNaNInf=false)
            val M_R_FP = Bin2FPCvt.FPAnyBinToFloat(FPBin=M_R_FPBin, ExpoWidth=4, MantWidth=4, CustomBias=Some(3), WithNaNInf=false)
            val UnderFlow = dut.io.M_W8_UnderFlow.toInt.toBinaryString
            val L_Part = s"[L] A (${A_6b_L_FPBin} = ${A_6b_L_FP}) * W (${W_6b_L_FPBin} = ${W_6b_L_FP})  =  [DUT] M (${M_L_FPBin} = ${M_L_FP})"
            val R_Part = s"[R] A (${A_6b_R_FPBin} = ${A_6b_R_FP}) * W (${W_6b_R_FPBin} = ${W_6b_R_FP})  =  [DUT] M (${M_R_FPBin} = ${M_R_FP})"
            // === Golden Model Execution ===
            val RealProd = A_6b_R_FP * W_6b_R_FP
            val RealE4M4 = FP2BinCvt.FloatToFPAnyBin(RealProd, ExpoWidth=4, MantWidth=4, CustomBias=Some(3), withNaNInf=false)
            // === Compare & Print ===
            val Compare = TU.checkResult(M_R_FP == RealProd)
            val A_str = s"A: ${A_6b_R_FPBin} = %-6s".format(A_6b_R_FP)
            val W_str = s"W: ${'0'+W_6b_R_FPBin} = %-6s".format(W_6b_R_FP)
            val Col_Input  = s"[Input L/R] $A_str  *  $W_str"
            val Col_DUT_L  = s"[DUT Ch0] M: ${M_L_FPBin} = %-8s".format(M_L_FP)
            val Col_DUT_R  = s"[DUT Ch1] M: ${M_R_FPBin} = %-8s".format(M_R_FP)
            val Col_Golden = s"[Golden Approx] M: ${RealE4M4} = %-10s".format(RealProd)
            val Col_Check  = s"[Check] $Compare"
            printf(s"%-56s | %-30s | %-30s | %-34s | %s\n", Col_Input, Col_DUT_L, Col_DUT_R, Col_Golden, Col_Check)
          }
        }
      } else  {    // MARK: W8A8
        for (clk <- 0 until 200) {
          // test case
          if (clk >= 10 && clk < 128+10) {
            // FP6 & FP8
            dut.io.A_6b_L  #= (clk - 10) / 64
            dut.io.A_6b_R  #= (clk - 10) % 64
            dut.io.W_6b_L  #= (clk - 10) / 64
            dut.io.W_6b_R  #= (clk - 10) % 64
            dut.io.PE_Mode #= PE_Mode
          } else {
            dut.io.A_6b_L  #= 0
            dut.io.A_6b_R  #= 0
            dut.io.W_6b_L  #= 0
            dut.io.W_6b_R  #= 0
            dut.io.PE_Mode #= PE_Mode
          }
          dut.clockDomain.waitRisingEdge()    // sample on rising edge

          if (clk >= 10 && clk < 128+10) {
            // * Print result
            // FP6 & FP8
            // A
            val A_6b_L_Bin = String.format(s"%6s", dut.io.A_6b_L.toInt.toBinaryString).replace(' ', '0')
            val A_6b_R_Bin = String.format(s"%6s", dut.io.A_6b_R.toInt.toBinaryString).replace(' ', '0')
            val A_E4M3_Bin = A_6b_L_Bin.substring(5, 6) + A_6b_R_Bin
            val A_E4M3_FP  = Bin2FPCvt.FPAnyBinToFloat(FPBin='0'+A_E4M3_Bin, ExpoWidth=4, MantWidth=3, CustomBias=Some(7), WithNaNInf=false)
            // W
            val W_6b_L_Bin = String.format(s"%6s", dut.io.W_6b_L.toInt.toBinaryString).replace(' ', '0')
            val W_6b_R_Bin = String.format(s"%6s", dut.io.W_6b_R.toInt.toBinaryString).replace(' ', '0')
            val W_E4M3_Bin = W_6b_L_Bin.substring(5, 6) + W_6b_R_Bin
            val W_E4M3_FP  = Bin2FPCvt.FPAnyBinToFloat(FPBin='0'+W_E4M3_Bin, ExpoWidth=4, MantWidth=3, CustomBias=Some(7), WithNaNInf=false)
            // M
            val M_A_FPBin = String.format(s"%10s", dut.io.M_S1E5M4_B7_HR.toInt.toBinaryString).replace(' ', '0')
            val M_A_FP    = Bin2FPCvt.FPAnyBinToFloat(FPBin=M_A_FPBin, ExpoWidth=5, MantWidth=4, CustomBias=Some(7), WithNaNInf=false)
            // === Golden Model Execution ===
            val A_8b_BigInt = BigInt("0" + A_E4M3_Bin, 2)
            val W_8b_BigInt = BigInt("0" + W_E4M3_Bin, 2)
            // Golden Model Result
            val Golden_Bin  = SFPMA_8b.calculate(W_8b_BigInt, A_8b_BigInt)
            val Golden_FP   = Bin2FPCvt.FPAnyBinToFloat(FPBin=Golden_Bin, ExpoWidth=5, MantWidth=4, CustomBias=Some(7), WithNaNInf=false)
            // === Compare & Print ===
            val Compare = TU.checkResult(M_A_FP == Golden_FP)
            val UF_Tag = if (dut.io.M_W8_UnderFlow.toBoolean) s"\u001B[35m[UF]\u001B[0m " else "     "
            val A_str = s"A: ${'0'+A_E4M3_Bin} = %-11s".format(A_E4M3_FP)
            val W_str = s"W: ${'0'+W_E4M3_Bin} = %-11s".format(W_E4M3_FP)
            val Col_Input  = s"[Input] $A_str  *  $W_str"
            val Col_DUT    = s"[DUT Output] M: ${M_A_FPBin} = %-14s".format(M_A_FP)
            val Col_Golden = s"[Golden Approx] M: ${Golden_Bin} = %-14s".format(Golden_FP)
            val Col_Check  = s"[Check] $Compare"
            printf(s"%-68s | %-36s | %-36s | %s%s\n", Col_Input, Col_DUT, Col_Golden, UF_Tag, Col_Check)
          }
        }
      }
      sleep(50)
    }

  }

}