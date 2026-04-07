package UniCore.Convert

import spinal.core._
import spinal.core.sim._
import UniCore.Config
import scala.language.postfixOps


case class CvtMultiFormatW_4b8b() extends Component {
  val io = new Bundle {
    val W_in              = in  Bits(8 bits)
    val DynFP3FP4FmtSel_L = in  Bits(3 bits)      // [DynFP3]: E2M0I, E2M0, E1M1I, E1M1; [DynFP4]: E1M2I, E1M2, E2M1, E3M0
    val DynFP3FP4FmtSel_R = in  Bits(3 bits)      // [DynFP3]: E2M0I, E2M0, E1M1I, E1M1; [DynFP4]: E1M2I, E1M2, E2M1, E3M0
    val Z_E3M2_L          = in  Bits(5 bits)
    val Z_E3M2_R          = in  Bits(5 bits)
    val FP6FP8FmtSel      = in  Bool()            // MARK: 0 for FP6, 1 for FP8
    val PE_Mode           = in  Bool()            // MARK: 0 for 3/4bits-PE, 1 for 6/8bits-PE

    val W_6b_L            = out Bits(6 bits)      // S1E3M2
    val W_6b_R            = out Bits(6 bits)      // S1E3M2
  }
  noIoPrefix()

  // * Extracting
  // FP6
  val W_FP6_Sign = io.W_in(5)
  val W_FP6_E3M2 = io.W_in(4 downto 0)
  // FP8
  val W_FP8_Sign = io.W_in(7)
  val W_FP8_E4M3 = io.W_in(6 downto 0)

  // * Converting
  val CvtDynFP3FP4_L = new DynFP3FP4toS1E3M2()
  val CvtDynFP3FP4_R = new DynFP3FP4toS1E3M2()
  val CvtE3M2toE4M3  = new E3M2B3toE4M3B7()

  CvtDynFP3FP4_L.io.W_DynFP3FP4 := io.W_in(7 downto 4)
  CvtDynFP3FP4_L.io.DynFP3FP4FmtSel := io.DynFP3FP4FmtSel_L
  CvtDynFP3FP4_L.io.Z_E3M2 := io.Z_E3M2_L

  CvtDynFP3FP4_R.io.W_DynFP3FP4 := io.W_in(3 downto 0)
  CvtDynFP3FP4_R.io.DynFP3FP4FmtSel := io.DynFP3FP4FmtSel_R
  CvtDynFP3FP4_R.io.Z_E3M2 := io.Z_E3M2_R

  CvtE3M2toE4M3.io.E3M2_in := W_FP6_E3M2

  // * Selecting output
  val W_FP6FP8_Sign = Mux(io.FP6FP8FmtSel, W_FP8_Sign, W_FP6_Sign)
  val W_FP6FP8_E4M3 = Mux(io.FP6FP8FmtSel, W_FP8_E4M3, CvtE3M2toE4M3.io.E4M3_out)
  val W_FP6FP8_12b = W_FP6FP8_Sign ## B"000" ## W_FP6FP8_E4M3(6 downto 5) ## B"0" ## W_FP6FP8_E4M3(4 downto 0)

  // * Final Output of Conversion
  io.W_6b_L := Mux(io.PE_Mode, W_FP6FP8_12b(11 downto 6), CvtDynFP3FP4_L.io.Unified_S1E3M2)
  io.W_6b_R := Mux(io.PE_Mode, W_FP6FP8_12b(5  downto 0), CvtDynFP3FP4_R.io.Unified_S1E3M2)

}


object CvtMultiFormatW_4b8b_Gen extends App {
  Config.setGenSubDir("/UniCore")
  Config.spinal.generateVerilog(CvtMultiFormatW_4b8b()).printRtl().mergeRTLSource()
}