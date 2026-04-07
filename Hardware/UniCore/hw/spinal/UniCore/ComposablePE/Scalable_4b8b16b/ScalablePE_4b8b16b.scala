package UniCore.ComposablePE.Scalable_4b8b16b

import spinal.core._
import spinal.core.sim._
import UniCore.Config
import scala.language.postfixOps
import UniCore.Convert.{Bin2FPCvt, FP2BinCvt}


// * 3/4b Mode: [0] [1] [2] [3], with Comp_M2_US
// * 6/8b Mode: [0 , 1] [2 , 3], with Comp_M3_DS & Comp_M3_US
// * 16b  Mode: [0 , 1 , 2 , 3], With Comp_M10_DS
case class ScalablePE_4b8b16b(AccuWidth: Int=21, DebugMode: Boolean=false) extends Component {
  val io = new Bundle {
    val W_6b           = in  Vec(Bits(6 bits), 4)            // 3/4b Mode: S1E3M2 (B=3)
    val A_6b_CIN       = in  Vec(Bits(6 bits), 4)            // 3/4b Mode: S1E3M2 (B=3)
    val NegB           = in  Vec(Bits(5 bits), 4)            // 3/4b Mode: "10100", 6/8b Mode: "11110_01000", 16b Mode:
    val PE_Mode        = in  Bits(2 bits)                    // MARK: "00"/"01" for 3/4bits-PE, "10" for 6/8bits-PE, "11" for 16bits-PE

    val PSumIn         = in  Vec(SInt(AccuWidth bits), 4)

    val M_S1E4M4_B3_HR = out (Vec(Bits(9  bits), 4)).simPublic()
    val M_S1E5M4_B7_HR = out (Vec(Bits(10 bits), 2)).simPublic()
    val FP8_UnderFlow  = out (Vec(Bool(), 2)).simPublic()
    val M_S1E5M10_B15  = out (Bits(16 bits)).simPublic()
    val FP16_UnderFlow = out (Bool()).simPublic()

    val A_6b_COUT      = out Vec(Bits(6 bits), 4)
    val PSumOut        = out Vec(SInt(AccuWidth bits), 4)
  }
  noIoPrefix()

  // * Extracting
  val A_Sign = Vec(io.A_6b_CIN.map(_(5)))
  val W_Sign = Vec(io.W_6b.map(_(5)))

  // * Scalable FPMA
  val S_FPMA = new ScalableFPMA_4b8b16b()
  S_FPMA.io.W_5b := Vec(io.W_6b.map(_(4 downto 0)))
  S_FPMA.io.A_5b := Vec(io.A_6b_CIN.map(_(4 downto 0)))
  S_FPMA.io.NegB := io.NegB
  S_FPMA.io.PE_Mode := io.PE_Mode

  // * Sign
  val Sign_M_4b  = A_Sign ^ W_Sign
  val Sign_M_8b  = (0 until 2).map{ i => Sign_M_4b(i*2) }
  val Sign_M_16b = Sign_M_4b(0)

  // * FPMA Result
  (0 until 4).foreach{ i => io.M_S1E4M4_B3_HR(i) := Sign_M_4b(i) ## S_FPMA.io.M_E4M4_B3_HR(i) }
  (0 until 2).foreach{ i => io.M_S1E5M4_B7_HR(i) := Sign_M_8b(i) ## S_FPMA.io.M_E5M4_B7_HR(i) }
  io.M_S1E5M10_B15 := Sign_M_16b ## S_FPMA.io.M_E5M10_B15

  io.FP8_UnderFlow := S_FPMA.io.FP8_UnderFlow
  io.FP16_UnderFlow := S_FPMA.io.FP16_UnderFlow


  // * Zero Check
  // FP4
  val A_FP4_is_Zero = (0 until 4).map{ i => io.A_6b_CIN(i) === B"000000" }
  val W_FP4_is_Zero = (0 until 4).map{ i => io.W_6b(i)     === B"000000" }
  val M_FP4_is_Zero = (0 until 4).map{ i => A_FP4_is_Zero(i) | W_FP4_is_Zero(i) }
  // FP8
  val A_FP8_is_Zero = (0 until 2).map{ i => A_FP4_is_Zero(i*2) & A_FP4_is_Zero(i*2+1) }
  val W_FP8_is_Zero = (0 until 2).map{ i => W_FP4_is_Zero(i*2) & W_FP4_is_Zero(i*2+1) }
  val M_FP8_is_Zero = (0 until 2).map{ i => A_FP8_is_Zero(i) | W_FP8_is_Zero(i) }
  // FP16
  val A_FP16_is_Zero = A_FP4_is_Zero.reduce(_ & _)
  val W_FP16_is_Zero = W_FP4_is_Zero.reduce(_ & _)
  val M_FP16_is_Zero = A_FP16_is_Zero | W_FP16_is_Zero

  // * Prepare for Shifting
  // E4M4
  val M_FP4_E4 = (0 until 4).map{ i => S_FPMA.io.M_E4M4_B3_HR(i)(7 downto 4).asUInt }
  val M_FP4_is_Subnorm = (0 until 4).map{ i => M_FP4_E4(i) === U(0) }
  val M_FP4_M4 = (0 until 4).map{ i => Mux(M_FP4_is_Zero(i), B"0000", S_FPMA.io.M_E4M4_B3_HR(i)(3 downto 0)) }
  // E5M4
  val M_FP8_E5 = (0 until 2).map{ i => S_FPMA.io.M_E5M4_B7_HR(i)(8 downto 4).asUInt }
  val M_FP8_M4 = (0 until 2).map{ i => Mux(S_FPMA.io.FP8_UnderFlow(i) | M_FP8_is_Zero(i), B"0000", S_FPMA.io.M_E5M4_B7_HR(i)(3 downto 0)) }   // Masking FPMA underflow for FP8
  // E5M10
  val M_FP16_E5  = S_FPMA.io.M_E5M10_B15(14 downto 10).asUInt
  val M_FP16_M10 = Mux(S_FPMA.io.FP16_UnderFlow | M_FP16_is_Zero, B"0000000000", S_FPMA.io.M_E5M10_B15(9 downto 0))
  // Hidden 1
  val Hidden_1_4b  = (0 until 4).map{ i => (~(M_FP4_is_Zero(i) | M_FP4_is_Subnorm(i))) | S_FPMA.io.SP_ShouldMod(i) }
  val Hidden_1_8b  = (0 until 2).map{ i => ~(M_FP8_is_Zero(i) | S_FPMA.io.FP8_UnderFlow(i)) }
  val Hidden_1_16b = ~(M_FP16_is_Zero | S_FPMA.io.FP16_UnderFlow)
  // Normalized Product
  val Norm_Prod_4b  = (0 until 4).map{ i => Hidden_1_4b(i) ## M_FP4_M4(i) ## B"0" }    // Add one zero at the end to make it 6 bits
  val Norm_Prod_8b  = (0 until 2).map{ i => Hidden_1_8b(i) ## M_FP8_M4(i) ## B"0" }    // Add one zero at the end to make it 6 bits
  val Norm_Prod_16b = Hidden_1_16b ## M_FP16_M10


  // * Dual Shifter
  val E5_8b16b_0 = Mux(S_FPMA.io.is16bMode, M_FP16_E5, M_FP8_E5(0))
  val E5_8b16b_1 = Mux(S_FPMA.io.is16bMode, M_FP16_E5, M_FP8_E5(1))

  val Shift_8b16b_E5_0 = Mux(E5_8b16b_0 >= U(16), E5_8b16b_0-U(15), U(0))
  val Shift_8b16b_E5_1 = E5_8b16b_0
  val Shift_8b16b_E5_2 = Mux(E5_8b16b_1 >= U(16), E5_8b16b_1-U(15), U(0))
  val Shift_8b16b_E5_3 = E5_8b16b_1


  // Norm_16b: Split it into 6b pairs, then reuse the Norm_0~3
  val Norm_16b_L = B"0" ## Norm_Prod_16b(10 downto 6)
  val Norm_16b_R = Norm_Prod_16b(5 downto 0)
  // Norm_8b16b
  val Norm_8b16b_0 = Mux(S_FPMA.io.is16bMode, Norm_16b_L, Norm_Prod_8b(0))
  val Norm_8b16b_1 = Mux(S_FPMA.io.is16bMode, Norm_16b_R, Norm_Prod_8b(1))


  val Norm_0 = Mux(io.PE_Mode(1), B"000000" ## Norm_8b16b_0, B"0000" ## Norm_Prod_4b(0) ## B"00")    // Carefully adjusted to align the results,
  val Norm_1 = Mux(io.PE_Mode(1), Norm_8b16b_0 ## B"000000", B"0000" ## Norm_Prod_4b(1) ## B"00")    //   for better reusing adders.
  val Norm_2 = Mux(io.PE_Mode(1), B"000000" ## Norm_8b16b_1, B"0000" ## Norm_Prod_4b(2) ## B"00")    // Carefully adjusted to align the results,
  val Norm_3 = Mux(io.PE_Mode(1), Norm_8b16b_1 ## B"000000", B"0000" ## Norm_Prod_4b(3) ## B"00")    //   for better reusing adders.

  val Shift_0 = Mux(io.PE_Mode(1), Shift_8b16b_E5_0, M_FP4_E4(0).resize(5 bits))
  val Shift_1 = Mux(io.PE_Mode(1), Shift_8b16b_E5_1, M_FP4_E4(1).resize(5 bits))
  val Shift_2 = Mux(io.PE_Mode(1), Shift_8b16b_E5_2, M_FP4_E4(2).resize(5 bits))
  val Shift_3 = Mux(io.PE_Mode(1), Shift_8b16b_E5_3, M_FP4_E4(3).resize(5 bits))

  val Value_0 = Norm_0 << Shift_0
  val Value_1 = Norm_1 << Shift_1
  val Value_2 = Norm_2 << Shift_2
  val Value_3 = Norm_3 << Shift_3

  // 21 bits
  val Prod_0 = Value_0(26 downto 6)    // Here we pad to 21 bits for later addition,
  val Prod_1 = Value_1(26 downto 6)    //   while the actual values use less than 21 bits.
  val Prod_2 = Value_2(26 downto 6)    // Here we pad to 21 bits for later addition,
  val Prod_3 = Value_3(26 downto 6)    //   while the actual values use less than 21 bits.

  val Not4bMode = ~S_FPMA.io.is4bMode
  val LSB_0 = Prod_0(0) & (Not4bMode | ~M_FP4_is_Subnorm(0))    // Subnormal fixing
  val LSB_1 = Prod_1(0) & (Not4bMode | ~M_FP4_is_Subnorm(1))    // Subnormal fixing
  val LSB_2 = Prod_2(0) & (Not4bMode | ~M_FP4_is_Subnorm(2))    // Subnormal fixing
  val LSB_3 = Prod_3(0) & (Not4bMode | ~M_FP4_is_Subnorm(3))    // Subnormal fixing

  val Prod_0_Masked = Prod_0(20 downto 1) ## LSB_0
  val Prod_1_Masked = Prod_1(20 downto 1) ## LSB_1
  val Prod_2_Masked = Prod_2(20 downto 1) ## LSB_2
  val Prod_3_Masked = Prod_3(20 downto 1) ## LSB_3


  // * Only for debug
  if (DebugMode) {
    val Golden_Value_4b = (0 until 4).map{ i => Norm_Prod_4b(i) << M_FP4_E4(i) }
    val Golden_Prod_4b = (0 until 4).map{ i => Golden_Value_4b(i)(20 downto 4) }
    val Golden_Value_8b = (0 until 2).map{ i => Norm_Prod_8b(i) << M_FP8_E5(i) }
    val Golden_Value_8b_SplitL_0 = Golden_Value_8b(0)(35 downto 21)
    val Golden_Value_8b_SplitR_0 = Golden_Value_8b(0)(20 downto 0)
    val Golden_Value_8b_SplitL_1 = Golden_Value_8b(1)(35 downto 21)
    val Golden_Value_8b_SplitR_1 = Golden_Value_8b(1)(20 downto 0)
    val Golden_Value_16b = Norm_Prod_16b << M_FP16_E5
    val Prod_Join_01 = Prod_0 ## Prod_1    // 42b
    val Prod_Join_23 = Prod_2 ## Prod_3    // 42b
    val Prod_FP16_Whole = (Prod_Join_01.asUInt << U(6)) + Prod_Join_23.asUInt
  }


  // * Accumulating
  val Accu_Pair_01 = new Dual_Accumulator(Width=AccuWidth)
  val Accu_Pair_23 = new Dual_Accumulator(Width=AccuWidth)
  Accu_Pair_01.io.SMag_Sign_4b_L_in := Mux(S_FPMA.io.is4bMode, Sign_M_4b(0), False)
  Accu_Pair_01.io.SMag_Sign_4b_R_in := Mux(S_FPMA.io.is4bMode, Sign_M_4b(1), False)
  Accu_Pair_01.io.SMag_Sign_8b_in := Mux(S_FPMA.io.is16bMode, False, Sign_M_8b(0))
  Accu_Pair_01.io.SMag_Magn_L_in := Prod_0_Masked
  Accu_Pair_01.io.SMag_Magn_R_in := Prod_1_Masked

  Accu_Pair_01.io.SInt_L_in := io.PSumIn(0)
  Accu_Pair_01.io.SInt_R_in := io.PSumIn(1)
  Accu_Pair_01.io.PE_Mode := io.PE_Mode(1)

  Accu_Pair_23.io.SMag_Sign_4b_L_in := Mux(S_FPMA.io.is4bMode, Sign_M_4b(2), False)
  Accu_Pair_23.io.SMag_Sign_4b_R_in := Mux(S_FPMA.io.is4bMode, Sign_M_4b(3), False)
  Accu_Pair_23.io.SMag_Sign_8b_in := Mux(S_FPMA.io.is16bMode, Sign_M_16b, Sign_M_8b(1))
  Accu_Pair_23.io.SMag_Magn_L_in := Prod_2_Masked
  Accu_Pair_23.io.SMag_Magn_R_in := Prod_3_Masked

  Accu_Pair_23.io.SInt_L_in := io.PSumIn(2)
  Accu_Pair_23.io.SInt_R_in := io.PSumIn(3)
  Accu_Pair_23.io.PE_Mode := io.PE_Mode(1)


  // * Cascade Output
  io.A_6b_COUT := io.A_6b_CIN

  // * Output
  io.PSumOut(0) := Accu_Pair_01.io.SInt_L_out
  io.PSumOut(1) := Accu_Pair_01.io.SInt_R_out
  io.PSumOut(2) := Accu_Pair_23.io.SInt_L_out
  io.PSumOut(3) := Accu_Pair_23.io.SInt_R_out

}


object ScalablePE_4b8b16b_Gen extends App {
  Config.setGenSubDir("/UniCore")
  Config.spinal.generateVerilog(ScalablePE_4b8b16b()).printRtl().mergeRTLSource()
}