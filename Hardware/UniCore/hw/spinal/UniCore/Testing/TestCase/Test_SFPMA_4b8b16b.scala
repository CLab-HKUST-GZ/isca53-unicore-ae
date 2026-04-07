package UniCore.Testing.TestCase

import spinal.core._
import spinal.core.sim._
import UniCore.Config
import scala.language.postfixOps
import UniCore.ComposablePE.Scalable_4b8b16b.ScalableFPMA_4b8b16b
import UniCore.Convert.{Bin2FPCvt, FP2BinCvt}
import UniCore.Testing.GoldenModels.{SFPMA_8b, SFPMA_16b}


object Test_SFPMA_4b8b16b {

  def main(args: Array[String]): Unit = {
    runW4A4Mode()
    runW8A8Mode()
    runW16A16Mode()
  }

  def runAll(): Unit = {
    runW4A4Mode()
    runW8A8Mode()
    runW16A16Mode()
  }

  def runW4A4Mode(): Unit = {
    // ==========================================
    // W4A4 Mode
    // ==========================================
    println(s"\n${"="*98}")
    println(s"[Test] S-FPMA test in W4A4 Mode:")
    println(s"${"-"*98}")
    println(s"[*] NOTE:")
    println(s"    - Parallelism : 4 independent output channels [Ch0] to [Ch3].")
    println(s"    - Input A Fmt : FP4 (S1E2M1).")
    println(s"    - Input W Fmt : DynFP3 / DynFP4.")
    println(s"${"="*98}\n")
    runTest(PE_Mode=1)    // MARK: FP3/4
  }

  def runW8A8Mode(): Unit = {
    // ==========================================
    // W8A8 Mode
    // ==========================================
    println(s"\n${"="*98}")
    println(s"[Test] S-FPMA test in W8A8 Mode:")
    println(s"${"-"*98}")
    println(s"[*] NOTE:")
    println(s"    - Parallelism : 2 independent output channels [DUT Ch0] and [DUT Ch1].")
    println(s"    - Input A Fmt : FP8 (S1E4M3).")
    println(s"    - Input W Fmt : FP6 / FP8 (S1E4M3).")
    println(s"    - UnderFlow   : A [UF] tag will trigger the Flush-to-Zero (FTZ) protect logic in PE.")
    println(s"${"="*98}\n")
    runTest(PE_Mode=2)     // MARK: FP6/8
  }

  def runW16A16Mode(): Unit = {
    // ==========================================
    // W16A16 Mode
    // ==========================================
    println(s"\n${"="*98}")
    println(s"[Test] S-FPMA test in W16A16 Mode:")
    println(s"${"-"*98}")
    println(s"[*] NOTE:")
    println(s"    - Parallelism : 1 combined high-precision output channel.")
    println(s"    - Inputs Fmt  : FP16 (S1E5M10) for both A and W.")
    println(s"    - UnderFlow   : A [UF] tag will trigger the Flush-to-Zero (FTZ) protect logic in PE.")
    println(s"${"="*98}\n")
    runTest(PE_Mode=3)     // MARK: FP16
  }


  def runTest(PE_Mode: Int): Unit = {

    var A_Idx = 0
    var W_Idx = 0

    // * 3/4b Mode
    val A_5b_FPBin   = Array(s"0", s"0", s"0", s"0")
    val A_5b_FP      = Array(0.0, 0.0, 0.0, 0.0)
    val W_5b_FPBin   = Array(s"0", s"0", s"0", s"0")
    val W_5b_FP      = Array(0.0, 0.0, 0.0, 0.0)
    val M_E4M4_FPBin = Array(s"0", s"0", s"0", s"0")
    val M_E4M4_FP    = Array(0.0, 0.0, 0.0, 0.0)

    Config.sim.compile{ScalableFPMA_4b8b16b()}.doSim { dut =>
      // simulation process
      dut.clockDomain.forkStimulus(2)
      // simulation code
      if (PE_Mode == 1) {
        // * W4A4
        for (clk <- 0 until 200) {
          // test case
          if (clk >= 10 && clk < (TU.A4_ValueSpace.length * TU.W4_ValueSpace.length)+10) {
            // FP3 & FP4
            A_Idx = (clk - 10) / TU.W4_ValueSpace.length
            W_Idx = (clk - 10) % TU.W4_ValueSpace.length
            dut.io.A_5b.foreach{ _ #= TU.A4_ValueSpace(A_Idx) % 32 }    // Remove the Sign bit
            dut.io.W_5b.foreach{ _ #= TU.W4_ValueSpace(W_Idx) % 32 }    // Remove the Sign bit
            dut.io.NegB.foreach{ _ #= BigInt("10100", 2) }
            dut.io.PE_Mode #= PE_Mode
          } else {
            dut.io.A_5b.foreach{ _ #= 0 }
            dut.io.W_5b.foreach{ _ #= 0 }
            dut.io.NegB.foreach{ _ #= BigInt("10100", 2) }
            dut.io.PE_Mode #= PE_Mode
          }
          dut.clockDomain.waitRisingEdge()    // sample on rising edge

          // * Print result
          // FP3 & FP4
          if (clk >= 10 && clk < (TU.A4_ValueSpace.length * TU.W4_ValueSpace.length)+10) {
            // A
            (0 until 4).foreach{ i => A_5b_FPBin(i) = String.format(s"%5s", dut.io.A_5b(i).toInt.toBinaryString).replace(' ', '0') }    // E3M2_B3
            (0 until 4).foreach{ i => A_5b_FP(i) = Bin2FPCvt.FPAnyBinToFloat(FPBin='0'+A_5b_FPBin(i), ExpoWidth=3, MantWidth=2, CustomBias=Some(3), WithNaNInf=false) }
            // W
            (0 until 4).foreach{ i => W_5b_FPBin(i) = String.format(s"%5s", dut.io.W_5b(i).toInt.toBinaryString).replace(' ', '0') }    // E3M2_B3
            (0 until 4).foreach{ i => W_5b_FP(i) = Bin2FPCvt.FPAnyBinToFloat(FPBin='0'+W_5b_FPBin(i), ExpoWidth=3, MantWidth=2, CustomBias=Some(3), WithNaNInf=false) }
            // M
            (0 until 4).foreach{ i => M_E4M4_FPBin(i) = String.format(s"%8s", dut.io.M_E4M4_B3_HR(i).toInt.toBinaryString).replace(' ', '0') }
            (0 until 4).foreach{ i => M_E4M4_FP(i) = Bin2FPCvt.FPAnyBinToFloat(FPBin='0'+M_E4M4_FPBin(i), ExpoWidth=4, MantWidth=4, CustomBias=Some(3), WithNaNInf=false) }
            // Print
            val AW_Part = s"A (${A_5b_FPBin(0)} = ${A_5b_FP(0)}) * W (${W_5b_FPBin(0)} = ${W_5b_FP(0)})"
            val M_Part_0 = s"= [Ch0] M (${M_E4M4_FPBin(0)} = ${M_E4M4_FP(0)})  "
            val M_Part_1 = s"[Ch1] M (${M_E4M4_FPBin(1)} = ${M_E4M4_FP(1)})  "
            val M_Part_2 = s"[Ch2] M (${M_E4M4_FPBin(2)} = ${M_E4M4_FP(2)})  "
            val M_Part_3 = s"[Ch3] M (${M_E4M4_FPBin(3)} = ${M_E4M4_FP(3)})  "
            val RealProd = A_5b_FP(0) * W_5b_FP(0)
            val RealE4M4 = FP2BinCvt.FloatToFPAnyBin(RealProd, ExpoWidth=4, MantWidth=4, CustomBias=Some(3), withNaNInf=false)
            val Compare = TU.checkResult(M_E4M4_FP(0) == RealProd)
            val RealValue = s"[Golden] ${RealE4M4} = ${RealProd}"
            printf(s"%-40s%-32s%-30s%-30s%-30s%-30s | [Check] %s\n", AW_Part, M_Part_0, M_Part_1, M_Part_2, M_Part_3, RealValue, Compare)
          }
        }
      } else if (PE_Mode == 2) {
        // * W8A8
        for (clk <- 0 until 200) {
          // test case
          if (clk >= 10 && clk < 128+10) {
            // FP6 & FP8
            dut.io.A_5b(0) #= (clk - 10) / 32
            dut.io.A_5b(1) #= (clk - 10) % 32
            dut.io.A_5b(2) #= (clk - 10) / 32
            dut.io.A_5b(3) #= (clk - 10) % 32
            dut.io.W_5b(0) #= (clk - 10) / 32
            dut.io.W_5b(1) #= (clk - 10) % 32
            dut.io.W_5b(2) #= (clk - 10) / 32
            dut.io.W_5b(3) #= (clk - 10) % 32
            dut.io.NegB(0) #= BigInt("11110", 2)
            dut.io.NegB(1) #= BigInt("01000", 2)
            dut.io.NegB(2) #= BigInt("11110", 2)
            dut.io.NegB(3) #= BigInt("01000", 2)
            dut.io.PE_Mode #= PE_Mode
          } else {
            dut.io.A_5b(0) #= 0
            dut.io.A_5b(1) #= 0
            dut.io.A_5b(2) #= 0
            dut.io.A_5b(3) #= 0
            dut.io.W_5b(0) #= 0
            dut.io.W_5b(1) #= 0
            dut.io.W_5b(2) #= 0
            dut.io.W_5b(3) #= 0
            dut.io.NegB(0) #= BigInt("11110", 2)
            dut.io.NegB(1) #= BigInt("01000", 2)
            dut.io.NegB(2) #= BigInt("11110", 2)
            dut.io.NegB(3) #= BigInt("01000", 2)
            dut.io.PE_Mode #= PE_Mode
          }
          dut.clockDomain.waitRisingEdge()    // sample on rising edge

          if (clk >= 10 && clk < 128+10) {
            // * Print result
            // FP6 & FP8
            // A
            (0 until 4).foreach{ i => A_5b_FPBin(i) = String.format(s"%5s", dut.io.A_5b(i).toInt.toBinaryString).replace(' ', '0') }
            val A_E4M3_Bin_L = A_5b_FPBin(0).substring(3, 5) + A_5b_FPBin(1)
            val A_E4M3_Bin_R = A_5b_FPBin(2).substring(3, 5) + A_5b_FPBin(3)
            val A_E4M3_FP_L  = Bin2FPCvt.FPAnyBinToFloat(FPBin='0'+A_E4M3_Bin_L, ExpoWidth=4, MantWidth=3, CustomBias=Some(7), WithNaNInf=false)
            val A_E4M3_FP_R  = Bin2FPCvt.FPAnyBinToFloat(FPBin='0'+A_E4M3_Bin_R, ExpoWidth=4, MantWidth=3, CustomBias=Some(7), WithNaNInf=false)
            // W
            (0 until 4).foreach{ i => W_5b_FPBin(i) = String.format(s"%5s", dut.io.W_5b(i).toInt.toBinaryString).replace(' ', '0') }
            val W_E4M3_Bin_L = W_5b_FPBin(0).substring(3, 5) + W_5b_FPBin(1)
            val W_E4M3_Bin_R = W_5b_FPBin(2).substring(3, 5) + W_5b_FPBin(3)
            val W_E4M3_FP_L  = Bin2FPCvt.FPAnyBinToFloat(FPBin='0'+W_E4M3_Bin_L, ExpoWidth=4, MantWidth=3, CustomBias=Some(7), WithNaNInf=false)
            val W_E4M3_FP_R  = Bin2FPCvt.FPAnyBinToFloat(FPBin='0'+W_E4M3_Bin_R, ExpoWidth=4, MantWidth=3, CustomBias=Some(7), WithNaNInf=false)
            // M
            val M_A_FPBin_L = String.format(s"%9s", dut.io.M_E5M4_B7_HR(0).toInt.toBinaryString).replace(' ', '0')
            val M_A_FPBin_R = String.format(s"%9s", dut.io.M_E5M4_B7_HR(1).toInt.toBinaryString).replace(' ', '0')
            val M_A_FP_L    = Bin2FPCvt.FPAnyBinToFloat(FPBin='0'+M_A_FPBin_L, ExpoWidth=5, MantWidth=4, CustomBias=Some(7), WithNaNInf=false)
            val M_A_FP_R    = Bin2FPCvt.FPAnyBinToFloat(FPBin='0'+M_A_FPBin_R, ExpoWidth=5, MantWidth=4, CustomBias=Some(7), WithNaNInf=false)
            // === Golden Model Execution ===
            val A_8b_BigInt = BigInt("0"+A_E4M3_Bin_L, 2)
            val W_8b_BigInt = BigInt("0"+W_E4M3_Bin_L, 2)
            val Golden_Bin  = SFPMA_8b.calculate(W_8b_BigInt, A_8b_BigInt)
            val Golden_FP   = Bin2FPCvt.FPAnyBinToFloat(FPBin=Golden_Bin, ExpoWidth=5, MantWidth=4, CustomBias=Some(7), WithNaNInf=false)
            // === Compare & Print ===
            val Compare = TU.checkResult(M_A_FP_L == Golden_FP)
            val UF_Tag = if (dut.io.FP8_UnderFlow(0).toBoolean) s"\u001B[35m[UF]\u001B[0m " else "     "
            val A_str = s"A: ${'0'+A_E4M3_Bin_L} = %-11s".format(A_E4M3_FP_L)
            val W_str = s"W: ${'0'+W_E4M3_Bin_L} = %-11s".format(W_E4M3_FP_L)
            val Col_Input  = s"[Input L/R] $A_str  *  $W_str"
            val Col_DUT_L  = s"[DUT Ch0] M: ${'0'+M_A_FPBin_L} = %-12s".format(M_A_FP_L)
            val Col_DUT_R  = s"[DUT Ch1] M: ${'0'+M_A_FPBin_R} = %-12s".format(M_A_FP_R)
            val Col_Golden = s"[Golden Approx] M: ${Golden_Bin} = %-12s".format(Golden_FP)
            val Col_Check  = s"[Check] $Compare"
            printf(s"%-70s | %-35s | %-35s | %-35s | %s%s\n", Col_Input, Col_DUT_L, Col_DUT_R, Col_Golden, UF_Tag, Col_Check)
          }
        }
      } else if (PE_Mode == 3) {
        // * W16A16
        for (clk <- 0 until 120) {
          if (clk > 10 && clk < 10+101) {
            // BF16
            val FP16_FP = clk - 10
            val FP16_FPBin = FP2BinCvt.FloatToFPAnyBin(f=FP16_FP, ExpoWidth=5, MantWidth=10, CustomBias=None, withNaNInf=true)
            val FP16_FPBinAsInt = BigInt(FP16_FPBin.replace("_", ""), 2)
            // println(FP16_FP, FP16_FPBin, FP16_FPBinAsInt)
            dut.io.A_5b(0) #= (FP16_FPBinAsInt / 1024) / 32
            dut.io.A_5b(1) #= (FP16_FPBinAsInt / 1024) % 32
            dut.io.A_5b(2) #= (FP16_FPBinAsInt % 1024) / 32
            dut.io.A_5b(3) #= (FP16_FPBinAsInt % 1024) % 32
            dut.io.W_5b(0) #= (FP16_FPBinAsInt / 1024) / 32
            dut.io.W_5b(1) #= (FP16_FPBinAsInt / 1024) % 32
            dut.io.W_5b(2) #= (FP16_FPBinAsInt % 1024) / 32
            dut.io.W_5b(3) #= (FP16_FPBinAsInt % 1024) % 32
            // -B = -15, NegB = 11111_10001_00000_00000
            dut.io.NegB(0) #= BigInt("11111", 2)
            dut.io.NegB(1) #= BigInt("10001", 2)
            dut.io.NegB(2) #= BigInt("00000", 2)
            dut.io.NegB(3) #= BigInt("00000", 2)
            dut.io.PE_Mode #= PE_Mode
          } else {
            dut.io.A_5b(0) #= 0
            dut.io.A_5b(1) #= 0
            dut.io.A_5b(2) #= 0
            dut.io.A_5b(3) #= 0
            dut.io.W_5b(0) #= 0
            dut.io.W_5b(1) #= 0
            dut.io.W_5b(2) #= 0
            dut.io.W_5b(3) #= 0
            // -B = -15, NegB = 11111_10001_00000_00000
            dut.io.NegB(0) #= BigInt("11111", 2)
            dut.io.NegB(1) #= BigInt("10001", 2)
            dut.io.NegB(2) #= BigInt("00000", 2)
            dut.io.NegB(3) #= BigInt("00000", 2)
            dut.io.PE_Mode #= PE_Mode
          }
          dut.clockDomain.waitRisingEdge()    // sample on rising edge

          if (clk > 10 && clk < 10+101) {
            // * Print result
            // FP16
            // A
            (0 until 4).foreach{ i => A_5b_FPBin(i) = String.format(s"%5s", dut.io.A_5b(i).toInt.toBinaryString).replace(' ', '0') }
            val A_E5M10_Bin = A_5b_FPBin(1) + A_5b_FPBin(2) + A_5b_FPBin(3)
            val A_E5M10_FP  = Bin2FPCvt.FPAnyBinToFloat(FPBin='0'+A_E5M10_Bin, ExpoWidth=5, MantWidth=10, CustomBias=None, WithNaNInf=false)
            // W
            (0 until 4).foreach{ i => W_5b_FPBin(i) = String.format(s"%5s", dut.io.W_5b(i).toInt.toBinaryString).replace(' ', '0') }
            val W_E5M10_Bin = W_5b_FPBin(1) + W_5b_FPBin(2) + W_5b_FPBin(3)
            val W_E5M10_FP  = Bin2FPCvt.FPAnyBinToFloat(FPBin='0'+W_E5M10_Bin, ExpoWidth=5, MantWidth=10, CustomBias=None, WithNaNInf=false)
            // M
            val M_A_FPBin      = String.format(s"%15s", dut.io.M_E5M10_B15.toInt.toBinaryString).replace(' ', '0')
            val M_A_FP         = Bin2FPCvt.FPAnyBinToFloat(FPBin='0'+M_A_FPBin, ExpoWidth=5, MantWidth=10, CustomBias=None, WithNaNInf=false)
            val M_Golden_FPBin = FP2BinCvt.FloatToFPAnyBin(f=A_E5M10_FP*W_E5M10_FP, ExpoWidth=5, MantWidth=10, CustomBias=None, withNaNInf=true)
            // === Golden Model Execution ===
            val A_16b_BigInt = BigInt("0"+A_E5M10_Bin, 2)
            val W_16b_BigInt = BigInt("0"+W_E5M10_Bin, 2)
            val Golden_Bin   = SFPMA_16b.calculate(W_16b_BigInt, A_16b_BigInt)
            val Golden_FP    = Bin2FPCvt.FPAnyBinToFloat(FPBin=Golden_Bin, ExpoWidth=5, MantWidth=10, CustomBias=None, WithNaNInf=false)
            // === Compare & Print ===
            val Compare = TU.checkResult(M_A_FP == Golden_FP)
            val UF_Tag = if (dut.io.FP8_UnderFlow(0).toBoolean) s"\u001B[35m[UF]\u001B[0m " else "     "
            val A_str = s"A: ${'0'+A_E5M10_Bin} = %-5s".format(A_E5M10_FP)
            val W_str = s"W: ${'0'+W_E5M10_Bin} = %-5s".format(W_E5M10_FP)
            val Col_Input  = s"[Input] $A_str  * $W_str"
            val Col_DUT    = s"[DUT] M: ${'0'+M_A_FPBin} = %-10s".format(M_A_FP)
            val Col_Golden = s"[Golden Approx] M: ${Golden_Bin} = %-10s".format(Golden_FP)
            val Col_Check  = s"[Check] $Compare"
            printf(s"%-68s | %-36s | %-38s | %s%s\n", Col_Input, Col_DUT, Col_Golden, UF_Tag, Col_Check)
          }
        }
      }
      sleep(50)
    }


  }

}
