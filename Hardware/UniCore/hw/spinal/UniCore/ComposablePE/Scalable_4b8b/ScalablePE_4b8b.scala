package UniCore.ComposablePE.Scalable_4b8b

import spinal.core._
import spinal.core.sim._
import UniCore.Config
import scala.language.postfixOps
import UniCore.Compensation.{Comp_M3_DS, Comp_M3_US}


// MARK: W4A4 & W8A8
case class ScalablePE_4b8b(AccuWidth: Int=18, DebugMode: Boolean=false) extends Component {
  val io = new Bundle {
    val T_7b_CIN_L       = in  Bits(7 bits)     // Unified 7b T for L
    val T_7b_CIN_R       = in  Bits(7 bits)     // Unified 7b T for R
    val W_6b_L           = in  Bits(6 bits)     // 3/4b Mode: S1E3M2 (B=3)
    val W_6b_R           = in  Bits(6 bits)     // 3/4b Mode: S1E3M2 (B=3)

    val PE_Mode          = in  Bool()           // MARK: 0 for W4A4, 1 for W8A8

    val PSumIn_L         = in  SInt(AccuWidth bits)
    val PSumIn_R         = in  SInt(AccuWidth bits)

    // Debug outputs
    val M_S1E4M4_B3_HR_L = if(DebugMode) Some(out Bits(9  bits)) else None    // W4A4 S-FPMA Product
    val M_S1E4M4_B3_HR_R = if(DebugMode) Some(out Bits(9  bits)) else None    // W4A4 S-FPMA Product
    val M_S1E5M4_B7_HR   = if(DebugMode) Some(out Bits(10 bits)) else None    // W8A8 S-FPMA Product
    val M_W8_UnderFlow   = if(DebugMode) Some(out Bool())        else None    // W8A8 S-FPMA Product UnderFlow

    // Cascade output
    val T_7b_COUT_L      = out Bits(7 bits)     // Unified 7b T for L
    val T_7b_COUT_R      = out Bits(7 bits)     // Unified 7b T for R
    val PSumOut_L        = out SInt(AccuWidth bits)
    val PSumOut_R        = out SInt(AccuWidth bits)
  }
  noIoPrefix()


  // * Down-Sampling Compensation
  // W8A8
  val Comp_M3_DS_Join = new Comp_M3_DS()
  Comp_M3_DS_Join.io.X_MantMSB := io.T_7b_CIN_R(2 downto 0)
  Comp_M3_DS_Join.io.Y_MantMSB := io.W_6b_R(2 downto 0)
  // * Up-Scaling Compensation
  // W8A8
  val Comp_M3_US_Join = new Comp_M3_US()
  Comp_M3_US_Join.io.A_M3 := io.T_7b_CIN_R(2 downto 0)
  Comp_M3_US_Join.io.W_M3 := io.W_6b_R(2 downto 0)


  // * Scalable FPMA Stage 2
  val S_FPMA_UP = new ScalableFPMA_4b8b()
  S_FPMA_UP.io.T_7b_L := io.T_7b_CIN_L
  S_FPMA_UP.io.T_7b_R := io.T_7b_CIN_R
  S_FPMA_UP.io.W_6b_L := io.W_6b_L
  S_FPMA_UP.io.W_6b_R := io.W_6b_R
  S_FPMA_UP.io.PE_Mode := io.PE_Mode
  S_FPMA_UP.io.Comp_M3_DS_Val := Comp_M3_DS_Join.io.Comp
  S_FPMA_UP.io.Comp_M3_US_Val := Comp_M3_US_Join.io.HyperRes


  // * Special Modification (For W4A4 when W_E3M0=0.25 and A_E2M1=0.5)
  val SP_ShouldMod_L = (io.W_6b_L(4 downto 0) === B"00100") & (io.T_7b_CIN_L(5 downto 0) === B"111100")
  val SP_ShouldMod_R = (io.W_6b_R(4 downto 0) === B"00100") & (io.T_7b_CIN_R(5 downto 0) === B"111100")
  val Mod_SP_L = SP_ShouldMod_L | S_FPMA_UP.io.M_S1E4M4_B3_HR_L(3)
  val Mod_SP_R = SP_ShouldMod_R | S_FPMA_UP.io.M_S1E4M4_B3_HR_R(3)

  // * Product Checks
  val M_S1E4M4_B3_HR_L = S_FPMA_UP.io.M_S1E4M4_B3_HR_L(8 downto 4) ## Mod_SP_L ## S_FPMA_UP.io.M_S1E4M4_B3_HR_L(2 downto 0)
  val M_S1E4M4_B3_HR_R = S_FPMA_UP.io.M_S1E4M4_B3_HR_R(8 downto 4) ## Mod_SP_R ## S_FPMA_UP.io.M_S1E4M4_B3_HR_R(2 downto 0)
  val M_S1E5M4_B7_HR   = S_FPMA_UP.io.M_S1E5M4_B7_HR
  val Sign_M_4b_L = S_FPMA_UP.io.M_S1E4M4_B3_HR_L(8)
  val Sign_M_4b_R = S_FPMA_UP.io.M_S1E4M4_B3_HR_R(8)
  val Sign_M_8b = S_FPMA_UP.io.M_S1E5M4_B7_HR(9)

  if (DebugMode) {
    io.M_S1E4M4_B3_HR_L.get := M_S1E4M4_B3_HR_L
    io.M_S1E4M4_B3_HR_R.get := M_S1E4M4_B3_HR_R
    io.M_S1E5M4_B7_HR.get := M_S1E5M4_B7_HR
    io.M_W8_UnderFlow.get := S_FPMA_UP.io.FP8_UnderFlow
  }

  // * Zero Check
  // FP4
  val A_L_is_Zero = io.T_7b_CIN_L(5 downto 0) === B"110100"        // T = A - B
  val A_R_is_Zero = io.T_7b_CIN_R(5 downto 0) === B"110100"        // T = A - B
  val W_L_is_Zero = io.W_6b_L === B"000000"
  val W_R_is_Zero = io.W_6b_R === B"000000"
  val M_L_is_Zero = A_L_is_Zero | W_L_is_Zero
  val M_R_is_Zero = A_R_is_Zero | W_R_is_Zero
  // FP8
  val W_FP8_is_Zero = W_L_is_Zero & W_R_is_Zero
  val A_FP8_is_Zero = (io.T_7b_CIN_L(5 downto 0) === B"000011") & (io.T_7b_CIN_R(5 downto 0) === B"001000")
  val M_A_is_Zero = W_FP8_is_Zero | A_FP8_is_Zero


  // * Prepare for Shifting
  // E4M4
  val E4_L = M_S1E4M4_B3_HR_L(7 downto 4).asUInt
  val E4_R = M_S1E4M4_B3_HR_R(7 downto 4).asUInt
  val M_L_is_Subnorm = E4_L === U(0)
  val M_R_is_Subnorm = E4_R === U(0)
  val M4_L = Mux(M_L_is_Zero, B"0000", M_S1E4M4_B3_HR_L(3 downto 0))
  val M4_R = Mux(M_R_is_Zero, B"0000", M_S1E4M4_B3_HR_R(3 downto 0))
  // E5M4
  val E5 = M_S1E5M4_B7_HR(8 downto 4).asUInt
  val M4 = Mux(S_FPMA_UP.io.FP8_UnderFlow, B"0000", M_S1E5M4_B7_HR(3 downto 0))    // Masking FPMA underflow for FP8
  // Hidden 1
  val Hidden_1_4b_L = (~(M_L_is_Zero | M_L_is_Subnorm)) | SP_ShouldMod_L
  val Hidden_1_4b_R = (~(M_R_is_Zero | M_R_is_Subnorm)) | SP_ShouldMod_R
  val Hidden_1_8b   = ~(M_A_is_Zero | S_FPMA_UP.io.FP8_UnderFlow)    // Masking FPMA underflow for FP8
  // Normalized Product
  val Norm_Prod_4b_L = Hidden_1_4b_L ## M4_L
  val Norm_Prod_4b_R = Hidden_1_4b_R ## M4_R
  val Norm_Prod_8b   = Hidden_1_8b ## M4


  // * Dual Shifter
  val E5_L = Mux(E5 >= U(14), E5-U(13), U(0))
  val E5_R = E5

  val Norm_L = Mux(io.PE_Mode, B"00000" ## Norm_Prod_8b, B"000" ## Norm_Prod_4b_L ## B"00")    // Carefully adjusted to align the results,
  val Norm_R = Mux(io.PE_Mode, Norm_Prod_8b ## B"00000", B"000" ## Norm_Prod_4b_R ## B"00")    //   for better reusing adders.

  val Shift_L = Mux(io.PE_Mode, E5_L, E4_L.resize(5 bits))
  val Shift_R = Mux(io.PE_Mode, E5_R, E4_R.resize(5 bits))

  val Value_L = Norm_L << Shift_L
  val Value_R = Norm_R << Shift_R

  val Prod_L = Value_L(22 downto 5)    // Here we pad to 18 bits for later addition,
  val Prod_R = Value_R(22 downto 5)    //   while the actual values use less than 18 bits.

  val LSB_L = Prod_L(0) & (io.PE_Mode | ~M_L_is_Subnorm)    // Subnormal fixing
  val LSB_R = Prod_R(0) & (io.PE_Mode | ~M_R_is_Subnorm)    // Subnormal fixing
  val Prod_L_Masked = Prod_L(17 downto 1) ## LSB_L
  val Prod_R_Masked = Prod_R(17 downto 1) ## LSB_R


  // * Only for debug
  if (DebugMode) {
    val Golden_Value_4b_L = Norm_Prod_4b_L << E4_L
    val Golden_Value_4b_R = Norm_Prod_4b_R << E4_R
    val Golden_Prod_4b_L = Golden_Value_4b_L(19 downto 3)
    val Golden_Prod_4b_R = Golden_Value_4b_R(19 downto 3)

    val Golden_Value_8b = Norm_Prod_8b << E5
    val Golden_Value_8b_SplitL = Golden_Value_8b(35 downto 18)
    val Golden_Value_8b_SplitR = Golden_Value_8b(17 downto 0)

    val Prod_whole = Prod_L ## Prod_R
    val CheckResult = Golden_Value_8b.asUInt - Prod_whole.asUInt
  }


  // * Accumulating
  val Accu_Pair = new Dual_Accumulator(Width=AccuWidth)
  Accu_Pair.io.SMag_Sign_4b_L_in := Sign_M_4b_L
  Accu_Pair.io.SMag_Sign_4b_R_in := Sign_M_4b_R
  Accu_Pair.io.SMag_Sign_8b_in := Sign_M_8b
  Accu_Pair.io.SMag_Magn_L_in := Prod_L_Masked
  Accu_Pair.io.SMag_Magn_R_in := Prod_R_Masked

  Accu_Pair.io.SInt_L_in := io.PSumIn_L
  Accu_Pair.io.SInt_R_in := io.PSumIn_R
  Accu_Pair.io.PE_Mode := io.PE_Mode


  // * Cascade Output
  io.T_7b_COUT_L := io.T_7b_CIN_L
  io.T_7b_COUT_R := io.T_7b_CIN_R

  // * Output
  io.PSumOut_L := Accu_Pair.io.SInt_L_out
  io.PSumOut_R := Accu_Pair.io.SInt_R_out

}


object ScalablePE_4b8b_Gen extends App {
  Config.setGenSubDir("/UniCore")
  Config.spinal.generateVerilog(ScalablePE_4b8b()).printRtl().mergeRTLSource()
}