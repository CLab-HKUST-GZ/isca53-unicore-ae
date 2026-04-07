package UniCore.Convert

import spinal.core._
import spinal.core.sim._
import UniCore.Config
import scala.language.postfixOps


case class CvtMultiFormatA_4b8b() extends Component {
  val io = new Bundle {
    val A_in          = in  Bits(8 bits)
    val FP6FP8FmtSel  = in  Bool()             // MARK: 0 for FP6, 1 for FP8
    val PE_Mode       = in  Bool()             // MARK: 0 for 3/4bits-PE, 1 for 6/8bits-PE

    val A_6b_L        = out Bits(6 bits)       // S1E3M2
    val A_6b_R        = out Bits(6 bits)       // S1E3M2
  }
  noIoPrefix()

  // * Extracting
  // FP4
  val A_FP4_Sign_L = io.A_in(3)
  val A_FP4_E2M1_L = io.A_in(2 downto 0)
  val A_FP4_Sign_R = io.A_in(7)
  val A_FP4_E2M1_R = io.A_in(6 downto 4)
  // FP6
  val A_FP6_Sign = io.A_in(5)
  val A_FP6_E3M2 = io.A_in(4 downto 0)
  // FP8
  val A_FP8_Sign = io.A_in(7)
  val A_FP8_E4M3 = io.A_in(6 downto 0)

  // * Converting
  val CvtE2M1toE3M1_L = new E2M1B1toE3M1B3()
  val CvtE2M1toE3M1_R = new E2M1B1toE3M1B3()
  val CvtE3M2toE4M3   = new E3M2B3toE4M3B7()

  CvtE2M1toE3M1_L.io.E2M1_in := A_FP4_E2M1_L
  CvtE2M1toE3M1_R.io.E2M1_in := A_FP4_E2M1_R
  CvtE3M2toE4M3.io.E3M2_in := A_FP6_E3M2

  // * Selecting output
  val A_S1E3M1B3_L = A_FP4_Sign_L ## CvtE2M1toE3M1_L.io.E3M1_out ## B"0"
  val A_S1E3M1B3_R = A_FP4_Sign_R ## CvtE2M1toE3M1_R.io.E3M1_out ## B"0"
  val A_FP6FP8_Sign = Mux(io.FP6FP8FmtSel, A_FP8_Sign, A_FP6_Sign)
  val A_FP6FP8_E4M3 = Mux(io.FP6FP8FmtSel, A_FP8_E4M3, CvtE3M2toE4M3.io.E4M3_out)
  val A_12b = A_FP6FP8_Sign ## B"000" ## A_FP6FP8_E4M3(6 downto 5) ## B"0" ## A_FP6FP8_E4M3(4 downto 0)

  // * Final Output of Conversion
  io.A_6b_L := Mux(io.PE_Mode, A_12b(11 downto 6), A_S1E3M1B3_L)
  io.A_6b_R := Mux(io.PE_Mode, A_12b(5  downto 0), A_S1E3M1B3_R)

}


object CvtMultiFormatA_4b8b_Gen extends App {
  Config.setGenSubDir("/UniCore")
  Config.spinal.generateVerilog(CvtMultiFormatA_4b8b()).printRtl().mergeRTLSource()
}