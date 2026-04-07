package UniCore.Testing.TestCase

import spinal.core._
import spinal.core.sim._
import UniCore.Config
import scala.language.postfixOps
import UniCore.ComposablePE.Scalable_4b8b16b.ScalablePE_4b8b16b
import UniCore.Convert.{Bin2FPCvt, FP2BinCvt}
import UniCore.Testing.GoldenModels.{SFPMA_8b, SFPMA_16b}
import scala.math.pow


object Test_CPE_4b8b16b {

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
    println(s"\n${"="*72}")
    println(s"[Test] Composable PE (4b-8b-16b) test in W4A4 Mode:")
    println(s"${"-"*72}")
    println(s"[*] NOTE:")
    println(s"    - Target      : PE-level MAC (Multiply-Accumulate) output.")
    println(s"    - Equation    : PSumOut = A * W + PSumIn.")
    println(s"    - Fmt support : Covering range for A in FP4, W in DynFP3/4.")
    println(s"    - Parallelism : 4 independent channels [Ch0] to [Ch3].")
    println(s"    - PSum Format : Fixed-point representation with 4 fractional bits.")
    println(s"${"="*72}\n")
    runTest(PE_Mode=1)    // MARK: FP3/4
  }

  def runW8A8Mode(): Unit = {
    // ==========================================
    // W8A8 Mode
    // ==========================================
    println(s"\n${"="*80}")
    println(s"[Test] Composable PE (4b-8b-16b) test in W8A8 Mode:")
    println(s"${"-"*80}")
    println(s"[*] NOTE:")
    println(s"    - Target      : PE-level MAC output.")
    println(s"    - Equation    : PSumOut = A * W + PSumIn.")
    println(s"    - Fmt support : Covering range for A in FP6/8, W in FP6/8.")
    println(s"    - Parallelism : 2 independent channels [Ch0] & [Ch1].")
    println(s"    - PSum Format : Fixed-point representation with 12 fractional bits.")
    println(s"    - UnderFlow   : A [UF] tag triggers the Flush-to-Zero (FTZ) protect.")
    println(s"${"="*80}\n")
    runTest(PE_Mode=2)     // MARK: FP6/8
  }

  def runW16A16Mode(): Unit = {
    // ==========================================
    // W16A16 Mode
    // ==========================================
    println(s"\n${"="*80}")
    println(s"[Test] Composable PE (4b-8b-16b) test in W16A16 Mode:")
    println(s"${"-"*80}")
    println(s"[*] NOTE:")
    println(s"    - Target      : PE-level MAC output.")
    println(s"    - Equation    : PSumOut = A * W + PSumIn.")
    println(s"    - Parallelism : 1 combined high-precision channel (84-bit Accu).")
    println(s"    - PSum Format : Fixed-point representation with 25 fractional bits.")
    println(s"    - UnderFlow   : A [UF] tag triggers the Flush-to-Zero (FTZ) protect.")
    println(s"${"="*80}\n")
    runTest(PE_Mode=3)     // MARK: FP16
  }


  def runTest(PE_Mode: Int): Unit = {

    // Test params
    var PSumIn_W4A4   = 100
    var PSumIn_W8A8   = 100
    var PSumIn_W16A16 = 100

    // W8A8 PSumIn Preprocessing (42b -> Two 21b SInts)
    val PSumIn_W8A8_42b = PSumIn_W8A8.toLong << 12    // fractional bits = 12
    val Mask21 = (1L << 21) - 1L                      // AccuWidth = 21
    val Lower_21b = PSumIn_W8A8_42b & Mask21
    val Upper_21b = (PSumIn_W8A8_42b >> 21) & Mask21
    val Lower_SInt = if (Lower_21b >= 1048576L) Lower_21b - 2097152L else Lower_21b    // 2^20 = 1048576, 2^21 = 2097152
    val Upper_SInt = if (Upper_21b >= 1048576L) Upper_21b - 2097152L else Upper_21b

    // W16A16 PSumIn Preprocessing (Overlapped Accumulator Injection)
    val PSumIn_W16A16_25b = PSumIn_W16A16.toLong << 25    // fractional bits = 25
    val In16_3 = PSumIn_W16A16_25b & Mask21
    val In16_2 = (PSumIn_W16A16_25b >> 21) & Mask21
    val In16_0_SInt = 0L
    val In16_1_SInt = 0L
    val In16_2_SInt = if (In16_2 >= 1048576L) In16_2 - 2097152L else In16_2
    val In16_3_SInt = if (In16_3 >= 1048576L) In16_3 - 2097152L else In16_3

    // Temp variables
    var A_Idx = 0
    var W_Idx = 0
    val A_6b_FPBin   = Array(s"0", s"0", s"0", s"0")
    val A_6b_FP      = Array(0.0, 0.0, 0.0, 0.0)
    val W_6b_FPBin   = Array(s"0", s"0", s"0", s"0")
    val W_6b_FP      = Array(0.0, 0.0, 0.0, 0.0)
    val M_S1E4M4_FPBin = Array(s"0", s"0", s"0", s"0")
    val M_S1E4M4_FP    = Array(0.0, 0.0, 0.0, 0.0)


    // * Hardware Simulation
    Config.sim.compile{ScalablePE_4b8b16b()}.doSim { dut =>
      dut.clockDomain.forkStimulus(2)
      if (PE_Mode == 1) {
        // * 3/4bits-PE
        for (clk <- 0 until 400) {
          // test case
          if (clk >= 10 && clk < (TU.A4_ValueSpace.length * TU.W4_ValueSpace.length)+10) {
            // FP3 & FP4
            A_Idx = (clk - 10) / TU.W4_ValueSpace.length
            W_Idx = (clk - 10) % TU.W4_ValueSpace.length
            dut.io.A_6b_CIN.foreach{ _ #= TU.A4_ValueSpace(A_Idx) }
            dut.io.W_6b.foreach{ _ #= TU.W4_ValueSpace(W_Idx) }
            dut.io.NegB.foreach{ _ #= BigInt("10100", 2) }
            dut.io.PE_Mode #= PE_Mode
            dut.io.PSumIn.foreach{ _ #= PSumIn_W4A4 << 4 }
          } else {
            dut.io.A_6b_CIN.foreach{ _ #= 0 }
            dut.io.W_6b.foreach{ _ #= 0 }
            dut.io.NegB.foreach{ _ #= BigInt("10100", 2) }
            dut.io.PE_Mode #= PE_Mode
            dut.io.PSumIn.foreach{ _ #= 0 }
          }
          dut.clockDomain.waitRisingEdge()    // sample on rising edge

          // * Print result
          // DynFP3 & DynFP4
          if (clk >= 10 && clk < (TU.A4_ValueSpace.length * TU.W4_ValueSpace.length)+10) {
            // A
            (0 until 4).foreach{ i => A_6b_FPBin(i) = String.format(s"%6s", dut.io.A_6b_CIN(i).toInt.toBinaryString).replace(' ', '0') }    // E3M2_B3
            (0 until 4).foreach{ i => A_6b_FP(i) = Bin2FPCvt.FPAnyBinToFloat(FPBin=A_6b_FPBin(i), ExpoWidth=3, MantWidth=2, CustomBias=Some(3), WithNaNInf=false) }
            // W
            (0 until 4).foreach{ i => W_6b_FPBin(i) = String.format(s"%6s", dut.io.W_6b(i).toInt.toBinaryString).replace(' ', '0') }    // E3M2_B3
            (0 until 4).foreach{ i => W_6b_FP(i) = Bin2FPCvt.FPAnyBinToFloat(FPBin=W_6b_FPBin(i), ExpoWidth=3, MantWidth=2, CustomBias=Some(3), WithNaNInf=false) }
            // M
            (0 until 4).foreach{ i => M_S1E4M4_FPBin(i) = String.format(s"%9s", dut.io.M_S1E4M4_B3_HR(i).toInt.toBinaryString).replace(' ', '0') }
            (0 until 4).foreach{ i => M_S1E4M4_FP(i) = Bin2FPCvt.FPAnyBinToFloat(FPBin=M_S1E4M4_FPBin(i), ExpoWidth=4, MantWidth=4, CustomBias=Some(3), WithNaNInf=false) }
            // PSum
            val PSum_0 = dut.io.PSumOut(0).toInt.toFloat / 16
            val PSum_1 = dut.io.PSumOut(1).toInt.toFloat / 16
            val PSum_2 = dut.io.PSumOut(2).toInt.toFloat / 16
            val PSum_3 = dut.io.PSumOut(3).toInt.toFloat / 16
            // === Golden Model Execution ===
            val RealProd = A_6b_FP(0) * W_6b_FP(0)
            val GoldenPSum = RealProd + PSumIn_W4A4
            // === Compare & Print ===
            val Compare = TU.checkResult(PSum_0 == GoldenPSum && PSum_1 == GoldenPSum && PSum_2 == GoldenPSum && PSum_3 == GoldenPSum)
            val A_str   = s"A: %-5s".format(A_6b_FP(0))
            val W_str   = s"W: %-5s".format(W_6b_FP(0))
            val PIn_str = s"PIn: %-4s".format(PSumIn_W4A4.toFloat)
            val DUT_str = s"[DUT Ch0-Ch3] PSum = %-8s, %-8s, %-8s, %-8s".format(PSum_0, PSum_1, PSum_2, PSum_3)
            val GLD_str = s"[Golden] PSum=%-9s".format(GoldenPSum)
            val Block = s"[W4A4] $A_str * $W_str + $PIn_str  =>  %-42s  vs  $GLD_str".format(DUT_str)
            printf(s"%-115s | %s\n", Block, s"[Check] $Compare")
          }
        }
      } else if (PE_Mode == 2) {
        // * 6/8bits-PE
        for (clk <- 0 until 200) {
          // test case
          if (clk >= 10 && clk < 128+10) {
            // FP6 & FP8
            dut.io.A_6b_CIN(0) #= (clk - 10) / 32
            dut.io.A_6b_CIN(1) #= (clk - 10) % 32
            dut.io.A_6b_CIN(2) #= (clk - 10) / 32
            dut.io.A_6b_CIN(3) #= (clk - 10) % 32
            dut.io.W_6b(0) #= (clk - 10) / 32
            dut.io.W_6b(1) #= (clk - 10) % 32
            dut.io.W_6b(2) #= (clk - 10) / 32
            dut.io.W_6b(3) #= (clk - 10) % 32
            dut.io.NegB(0) #= BigInt("11110", 2)
            dut.io.NegB(1) #= BigInt("01000", 2)
            dut.io.NegB(2) #= BigInt("11110", 2)
            dut.io.NegB(3) #= BigInt("01000", 2)
            dut.io.PE_Mode #= PE_Mode
            // dut.io.PSumIn.foreach{ _ #= 0 }
            dut.io.PSumIn(0) #= Upper_SInt
            dut.io.PSumIn(1) #= Lower_SInt
            dut.io.PSumIn(2) #= Upper_SInt
            dut.io.PSumIn(3) #= Lower_SInt
          } else {
            dut.io.A_6b_CIN(0) #= 0
            dut.io.A_6b_CIN(1) #= 0
            dut.io.A_6b_CIN(2) #= 0
            dut.io.A_6b_CIN(3) #= 0
            dut.io.W_6b(0) #= 0
            dut.io.W_6b(1) #= 0
            dut.io.W_6b(2) #= 0
            dut.io.W_6b(3) #= 0
            dut.io.NegB(0) #= BigInt("11110", 2)
            dut.io.NegB(1) #= BigInt("01000", 2)
            dut.io.NegB(2) #= BigInt("11110", 2)
            dut.io.NegB(3) #= BigInt("01000", 2)
            dut.io.PE_Mode #= PE_Mode
            dut.io.PSumIn.foreach{ _ #= 0 }
          }
          dut.clockDomain.waitRisingEdge()    // sample on rising edge

          if (clk >= 10 && clk < 128+10) {
            // * Print result
            // FP6 & FP8
            // === A ===
            (0 until 4).foreach{ i => A_6b_FPBin(i) = String.format(s"%6s", dut.io.A_6b_CIN(i).toInt.toBinaryString).replace(' ', '0') }
            val A_E4M3_Bin_L = A_6b_FPBin(0).substring(4, 6) + A_6b_FPBin(1).substring(1, 6)
            val A_E4M3_Bin_R = A_6b_FPBin(2).substring(4, 6) + A_6b_FPBin(3).substring(1, 6)
            val A_E4M3_FP_L  = Bin2FPCvt.FPAnyBinToFloat(FPBin='0'+A_E4M3_Bin_L, ExpoWidth=4, MantWidth=3, CustomBias=Some(7), WithNaNInf=false)
            val A_E4M3_FP_R  = Bin2FPCvt.FPAnyBinToFloat(FPBin='0'+A_E4M3_Bin_R, ExpoWidth=4, MantWidth=3, CustomBias=Some(7), WithNaNInf=false)
            // === W ===
            (0 until 4).foreach{ i => W_6b_FPBin(i) = String.format(s"%6s", dut.io.W_6b(i).toInt.toBinaryString).replace(' ', '0') }
            val W_E4M3_Bin_L = W_6b_FPBin(0).substring(4, 6) + W_6b_FPBin(1).substring(1, 6)
            val W_E4M3_Bin_R = W_6b_FPBin(2).substring(4, 6) + W_6b_FPBin(3).substring(1, 6)
            val W_E4M3_FP_L  = Bin2FPCvt.FPAnyBinToFloat(FPBin='0'+W_E4M3_Bin_L, ExpoWidth=4, MantWidth=3, CustomBias=Some(7), WithNaNInf=false)
            val W_E4M3_FP_R  = Bin2FPCvt.FPAnyBinToFloat(FPBin='0'+W_E4M3_Bin_R, ExpoWidth=4, MantWidth=3, CustomBias=Some(7), WithNaNInf=false)
            // M
            val M_A_FPBin_L = String.format(s"%10s", dut.io.M_S1E5M4_B7_HR(0).toInt.toBinaryString).replace(' ', '0')
            val M_A_FPBin_R = String.format(s"%10s", dut.io.M_S1E5M4_B7_HR(1).toInt.toBinaryString).replace(' ', '0')
            val M_A_FP_L    = Bin2FPCvt.FPAnyBinToFloat(FPBin=M_A_FPBin_L, ExpoWidth=5, MantWidth=4, CustomBias=Some(7), WithNaNInf=false, WithSubnorm=false)
            val M_A_FP_R    = Bin2FPCvt.FPAnyBinToFloat(FPBin=M_A_FPBin_R, ExpoWidth=5, MantWidth=4, CustomBias=Some(7), WithNaNInf=false, WithSubnorm=false)
            // === Golden Model Execution ===
            val A_8b_BigInt = BigInt("0"+A_E4M3_Bin_L, 2)
            val W_8b_BigInt = BigInt("0"+W_E4M3_Bin_L, 2)
            val Golden_M_Raw_Bin = SFPMA_8b.calculate(W_8b_BigInt, A_8b_BigInt)
            val Golden_M_Raw_FP  = Bin2FPCvt.FPAnyBinToFloat(FPBin=Golden_M_Raw_Bin, ExpoWidth=5, MantWidth=4, CustomBias=Some(7), WithNaNInf=false, WithSubnorm=false)
            val isUnderFlow_0 = dut.io.FP8_UnderFlow(0).toBoolean
            val isUnderFlow_1 = dut.io.FP8_UnderFlow(1).toBoolean
            val Golden_M_FP_0 = if (isUnderFlow_0) 0.0 else Golden_M_Raw_FP
            val Golden_M_FP_1 = if (isUnderFlow_1) 0.0 else Golden_M_Raw_FP
            // === [DUT] PSum Execution (42-bit Split & Reconstruct) ===
            val mask21_BigInt = (BigInt(1) << 21) - 1
            val PSum_0_H = BigInt(dut.io.PSumOut(0).toInt)
            val PSum_0_L = dut.io.PSumOut(1).toBigInt & mask21_BigInt
            val PSum_Ch0 = (PSum_0_H << 9).toDouble + (PSum_0_L.toDouble / 4096.0)
            val PSum_1_H = BigInt(dut.io.PSumOut(2).toInt)
            val PSum_1_L = dut.io.PSumOut(3).toBigInt & mask21_BigInt
            val PSum_Ch1 = (PSum_1_H << 9).toDouble + (PSum_1_L.toDouble / 4096.0)
            // === Golden PSum & Compare ===
            val GoldenPSum_0 = Golden_M_FP_0 + PSumIn_W8A8
            val GoldenPSum_1 = Golden_M_FP_1 + PSumIn_W8A8
            val Compare = TU.checkResult(PSum_Ch0 == GoldenPSum_0 && PSum_Ch1 == GoldenPSum_1)
            // === Format & Print ===
            val A_str   = s"A: %-11s".format(A_E4M3_FP_L)
            val W_str   = s"W: %-11s".format(W_E4M3_FP_L)
            val PIn_str = s"PIn: %-4s".format(PSumIn_W8A8.toDouble)
            val DUT_str = s"[DUT Ch0-1] PSum = %-16s, %-16s".format(PSum_Ch0, PSum_Ch1)
            val GLD_str = s"[GLD Approx] PSum=%-16s".format(GoldenPSum_0)
            val Block = s"[W8A8] $A_str * $W_str + $PIn_str  =>  %-42s vs  $GLD_str".format(DUT_str)
            // === Exact Math Execution ===
            val ExactPSum = (A_E4M3_FP_L * W_E4M3_FP_L) + PSumIn_W8A8
            val Exact_str = s"[Exact Math] PSum=%-18s".format(ExactPSum)
            val UF_Tag = if (isUnderFlow_0 || isUnderFlow_1) s"\u001B[35m[UF]\u001B[0m " else "     "
            printf(s"%-125s | %s%26s | %s\n", Block, UF_Tag, s"[Check] $Compare ", Exact_str)
          }
        }
      } else if (PE_Mode == 3) {
        // * 16bits-PE
        for (clk <- 0 until 120) {
          if (clk >= 10 && clk < 10+101) {
            // BF16
            val FP16_FP = clk - 10
            val FP16_FPBin = FP2BinCvt.FloatToFPAnyBin(f=FP16_FP, ExpoWidth=5, MantWidth=10, CustomBias=None, withNaNInf=true)
            val FP16_FPBinAsInt = BigInt(FP16_FPBin.replace("_", ""), 2)
            dut.io.A_6b_CIN(0) #= (FP16_FPBinAsInt / 1024) / 32
            dut.io.A_6b_CIN(1) #= (FP16_FPBinAsInt / 1024) % 32
            dut.io.A_6b_CIN(2) #= (FP16_FPBinAsInt % 1024) / 32
            dut.io.A_6b_CIN(3) #= (FP16_FPBinAsInt % 1024) % 32
            dut.io.W_6b(0) #= (FP16_FPBinAsInt / 1024) / 32
            dut.io.W_6b(1) #= (FP16_FPBinAsInt / 1024) % 32
            dut.io.W_6b(2) #= (FP16_FPBinAsInt % 1024) / 32
            dut.io.W_6b(3) #= (FP16_FPBinAsInt % 1024) % 32
            // -B = -15, NegB = 11111_10001_00000_00000
            dut.io.NegB(0) #= BigInt("11111", 2)
            dut.io.NegB(1) #= BigInt("10001", 2)
            dut.io.NegB(2) #= BigInt("00000", 2)
            dut.io.NegB(3) #= BigInt("00000", 2)
            dut.io.PE_Mode #= PE_Mode
            dut.io.PSumIn(0) #= In16_0_SInt
            dut.io.PSumIn(1) #= In16_1_SInt
            dut.io.PSumIn(2) #= In16_2_SInt
            dut.io.PSumIn(3) #= In16_3_SInt
          } else {
            dut.io.A_6b_CIN(0) #= 0
            dut.io.A_6b_CIN(1) #= 0
            dut.io.A_6b_CIN(2) #= 0
            dut.io.A_6b_CIN(3) #= 0
            dut.io.W_6b(0) #= 0
            dut.io.W_6b(1) #= 0
            dut.io.W_6b(2) #= 0
            dut.io.W_6b(3) #= 0
            // -B = -15, NegB = 11111_10001_00000_00000
            dut.io.NegB(0) #= BigInt("11111", 2)
            dut.io.NegB(1) #= BigInt("10001", 2)
            dut.io.NegB(2) #= BigInt("00000", 2)
            dut.io.NegB(3) #= BigInt("00000", 2)
            dut.io.PE_Mode #= PE_Mode
            dut.io.PSumIn.foreach{ _ #= 0 }
          }
          dut.clockDomain.waitRisingEdge()    // sample on rising edge

          if (clk >= 10 && clk < 10+101) {
            // * Print result
            // FP16
            // === A ===
            (0 until 4).foreach{ i => A_6b_FPBin(i) = String.format(s"%6s", dut.io.A_6b_CIN(i).toInt.toBinaryString).replace(' ', '0') }
            val A_16b_Bin = A_6b_FPBin(0).substring(5, 6) + A_6b_FPBin(1).substring(1, 6) + A_6b_FPBin(2).substring(1, 6) + A_6b_FPBin(3).substring(1, 6)
            val A_16b_FP  = Bin2FPCvt.FPAnyBinToFloat(FPBin=A_16b_Bin, ExpoWidth=5, MantWidth=10, CustomBias=None, WithNaNInf=false, WithSubnorm=false)
            // === W ===
            (0 until 4).foreach{ i => W_6b_FPBin(i) = String.format(s"%6s", dut.io.W_6b(i).toInt.toBinaryString).replace(' ', '0') }
            val W_16b_Bin = W_6b_FPBin(0).substring(5, 6) + W_6b_FPBin(1).substring(1, 6) + W_6b_FPBin(2).substring(1, 6) + W_6b_FPBin(3).substring(1, 6)
            val W_16b_FP  = Bin2FPCvt.FPAnyBinToFloat(FPBin=W_16b_Bin, ExpoWidth=5, MantWidth=10, CustomBias=None, WithNaNInf=false, WithSubnorm=false)
            // === M ===
            val M_A_FPBin = String.format(s"%16s", dut.io.M_S1E5M10_B15.toInt.toBinaryString).replace(' ', '0')
            val M_A_FP    = Bin2FPCvt.FPAnyBinToFloat(FPBin=M_A_FPBin, ExpoWidth=5, MantWidth=10, CustomBias=None, WithNaNInf=false, WithSubnorm=false)
            // === Golden Model Execution ===
            val A_16b_BigInt = BigInt(A_16b_Bin, 2)
            val W_16b_BigInt = BigInt(W_16b_Bin, 2)
            val Golden_M_Raw_Bin = SFPMA_16b.calculate(W_16b_BigInt, A_16b_BigInt)
            val Golden_M_Raw_FP  = Bin2FPCvt.FPAnyBinToFloat(FPBin=Golden_M_Raw_Bin, ExpoWidth=5, MantWidth=10, CustomBias=None, WithNaNInf=false, WithSubnorm=false)
            val isUnderFlow = dut.io.FP16_UnderFlow.toBoolean
            val Golden_M_FP = if (isUnderFlow) 0.0 else Golden_M_Raw_FP
            // === [DUT] PSum Execution (Overlapped Reconstruction) ===
            val PSumOut_0 = dut.io.PSumOut(0).toLong
            val PSumOut_1 = dut.io.PSumOut(1).toLong & 0x1FFFFFL
            val PSumOut_2 = dut.io.PSumOut(2).toLong
            val PSumOut_3 = dut.io.PSumOut(3).toLong & 0x1FFFFFL
            val Accu_01 = (PSumOut_0 << 21) | PSumOut_1    // Concat
            val Accu_23 = (PSumOut_2 << 21) | PSumOut_3    // Concat
            val PSum_Total_BigInt = (BigInt(Accu_01) << 6) + BigInt(Accu_23)
            val PSum = PSum_Total_BigInt.toDouble / pow(2, 25)
            // === Golden PSum & Compare ===
            val GoldenPSum = Golden_M_FP + PSumIn_W16A16
            val Compare = TU.checkResult(PSum == GoldenPSum)
            // === Format & Print ===
            val A_str   = s"A: %-11s".format(A_16b_FP)
            val W_str   = s"W: %-11s".format(W_16b_FP)
            val PIn_str = s"PIn: %-4s".format(PSumIn_W16A16.toDouble)
            val DUT_str = s"[DUT] PSum = %-12s".format(PSum)
            val GLD_str = s"[GLD Approx] PSum=%-12s".format(GoldenPSum)
            val Block = s"[W16A16] $A_str * $W_str + $PIn_str  =>  %-24s vs  $GLD_str".format(DUT_str)
            // === Exact Math Execution ===
            val ExactPSum = (A_16b_FP * W_16b_FP) + PSumIn_W16A16
            val Exact_str = s"[Exact Math] PSum=%-18s".format(ExactPSum)
            val UF_Tag = if (isUnderFlow) s"\u001B[35m[UF]\u001B[0m " else "     "
            printf(s"%-120s | %s%26s | %s\n", Block, UF_Tag, s"[Check] $Compare ", Exact_str)
          }

        }
      }
      sleep(50)
    }

  }


}
