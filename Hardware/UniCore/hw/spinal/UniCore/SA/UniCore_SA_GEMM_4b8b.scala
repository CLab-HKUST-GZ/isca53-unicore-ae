package UniCore.SA

import spinal.core._
import spinal.core.sim._
import UniCore.Config
import scala.language.postfixOps
import UniCore.ComposablePE.Scalable_4b8b.PreAdd_4b8b
import UniCore.Convert.{CvtMultiFormatA_4b8b, CvtMultiFormatW_4b8b}


// * UniCore SystolicArray for GEMM (Per-Group Quantization Supported, Systolic Spatial Array, Weight Stationary)
case class UniCore_SA_GEMM_4b8b(
                                 Slice_Num  : Int,
                                 Tile_Row   : Int,
                                 Tile_Col   : Int,
                                 DualPE_Row : Int,
                                 DualPE_Col : Int,
                                 AccuWidth  : Int
                               ) extends Component {
  val io = new Bundle {
    val DinTop_W_4b_L     = in  Vec(Vec(Vec(Bits(4 bits), DualPE_Col), Tile_Col), Slice_Num)    // W4
    val DinTop_W_4b_R     = in  Vec(Vec(Vec(Bits(4 bits), DualPE_Col), Tile_Col), Slice_Num)    // W4

    val DinTop_A_4b_L     = in  Vec(Vec(Vec(Bits(4 bits), DualPE_Col), Tile_Col), Slice_Num)    // A4
    val DinTop_A_4b_R     = in  Vec(Vec(Vec(Bits(4 bits), DualPE_Col), Tile_Col), Slice_Num)    // A4

    val PE_Mode           = in  Bool()                                                          // MARK: 0 for 3/4bits-PE, 1 for 6/8bits-PE
    val FP6FP8FmtSel_A    = in  Bool()                                                          // MARK: 0 for FP6, 1 for FP8
    val FP6FP8FmtSel_W    = in  Bool()                                                          // MARK: 0 for FP6, 1 for FP8
    val DynFP3FP4FmtSel_L = in  Vec(Vec(Vec(Bits(3 bits), DualPE_Col), Tile_Col), Slice_Num)
    val DynFP3FP4FmtSel_R = in  Vec(Vec(Vec(Bits(3 bits), DualPE_Col), Tile_Col), Slice_Num)
    val Z_E3M2_L          = in  Vec(Vec(Vec(Bits(5 bits), DualPE_Col), Tile_Col), Slice_Num)
    val Z_E3M2_R          = in  Vec(Vec(Vec(Bits(5 bits), DualPE_Col), Tile_Col), Slice_Num)
    val WqLock            = in  Bool()                                                          // Locking the preloaded Wq

    val PSumOut_L         = out Vec(Vec(Vec(SInt(AccuWidth bits), DualPE_Row), Tile_Row), Slice_Num)
    val PSumOut_R         = out Vec(Vec(Vec(SInt(AccuWidth bits), DualPE_Row), Tile_Row), Slice_Num)
  }
  noIoPrefix()

  // * Unified Format Converter (A)
  val Cvt_A_4b8b = (0 until Slice_Num).map{sl => (0 until Tile_Col).map{tc => (0 until DualPE_Col).map{dc =>
    new CvtMultiFormatA_4b8b().setName(s"CvtA_4b8b_sl${sl}_tc${tc}_dc${dc}")
  }}}

  // * Unified Format Converter (W)
  val Cvt_W_4b8b = (0 until Slice_Num).map{sl => (0 until Tile_Col).map{tc => (0 until DualPE_Col).map{dc =>
    new CvtMultiFormatW_4b8b().setName(s"CvtW_4b8b_sl${sl}_tc${tc}_dc${dc}")
  }}}

  // * Pre-Adder
  val PreAdder_4b8b = (0 until Slice_Num).map{sl => (0 until Tile_Col).map{tc => (0 until DualPE_Col).map{dc =>
    new PreAdd_4b8b().setName(s"PreAdd_4b8b_sl${sl}_tc${tc}_dc${dc}")
  }}}

  // * Connecting
  (0 until Slice_Num).foreach{sl => (0 until Tile_Col).foreach{tc => (0 until DualPE_Col).foreach{dc =>
    // Connection: CvtA
    Cvt_A_4b8b(sl)(tc)(dc).io.A_in := (
      io.DinTop_A_4b_L(sl)(tc)(dc).setName(s"DinTop_A_4b_L_sl${sl}_tc${tc}_dc${dc}") ##
      io.DinTop_A_4b_R(sl)(tc)(dc).setName(s"DinTop_A_4b_R_sl${sl}_tc${tc}_dc${dc}")
    )
    Cvt_A_4b8b(sl)(tc)(dc).io.PE_Mode := io.PE_Mode
    Cvt_A_4b8b(sl)(tc)(dc).io.FP6FP8FmtSel := io.FP6FP8FmtSel_A

    // Connection: CvtW
    Cvt_W_4b8b(sl)(tc)(dc).io.W_in := (
      io.DinTop_W_4b_L(sl)(tc)(dc).setName(s"DinTop_W_4b_L_sl${sl}_tc${tc}_dc${dc}") ##
      io.DinTop_W_4b_R(sl)(tc)(dc).setName(s"DinTop_W_4b_R_sl${sl}_tc${tc}_dc${dc}")
    )
    Cvt_W_4b8b(sl)(tc)(dc).io.PE_Mode := io.PE_Mode
    Cvt_W_4b8b(sl)(tc)(dc).io.FP6FP8FmtSel := io.FP6FP8FmtSel_W
    Cvt_W_4b8b(sl)(tc)(dc).io.DynFP3FP4FmtSel_L := io.DynFP3FP4FmtSel_L(sl)(tc)(dc).setName(s"DynFP3FP4FmtSel_L_sl${sl}_tc${tc}_dc${dc}")
    Cvt_W_4b8b(sl)(tc)(dc).io.DynFP3FP4FmtSel_R := io.DynFP3FP4FmtSel_R(sl)(tc)(dc).setName(s"DynFP3FP4FmtSel_R_sl${sl}_tc${tc}_dc${dc}")
    Cvt_W_4b8b(sl)(tc)(dc).io.Z_E3M2_L := io.Z_E3M2_L(sl)(tc)(dc).setName(s"Z_E3M2_L_sl${sl}_tc${tc}_dc${dc}")
    Cvt_W_4b8b(sl)(tc)(dc).io.Z_E3M2_R := io.Z_E3M2_R(sl)(tc)(dc).setName(s"Z_E3M2_R_sl${sl}_tc${tc}_dc${dc}")

    // Connection: PreAdd
    PreAdder_4b8b(sl)(tc)(dc).io.A_6b_L := Cvt_A_4b8b(sl)(tc)(dc).io.A_6b_L
    PreAdder_4b8b(sl)(tc)(dc).io.A_6b_R := Cvt_A_4b8b(sl)(tc)(dc).io.A_6b_R
    PreAdder_4b8b(sl)(tc)(dc).io.PE_Mode := io.PE_Mode
  }}}


  // * Generate Slices
  val Slices = (0 until Slice_Num).map{ sl => UniCore_Slice_4b8b(
    Tile_Row=Tile_Row, Tile_Col=Tile_Col, DualPE_Row=DualPE_Row, DualPE_Col=DualPE_Col, AccuWidth=AccuWidth
  ).setName(s"Slices_sl${sl}")}


  // * Vertical
  for (sl <- 0 until Slice_Num) {
    Slices(sl).io.PE_Mode := io.PE_Mode
    Slices(sl).io.WqLock := io.WqLock
    for (tc <- 0 until Tile_Col) {
      for (dc <- 0 until DualPE_Col) {
        Slices(sl).io.DinTop_W_6b_L(tc)(dc) := Cvt_W_4b8b(sl)(tc)(dc).io.W_6b_L
        Slices(sl).io.DinTop_W_6b_R(tc)(dc) := Cvt_W_4b8b(sl)(tc)(dc).io.W_6b_R
        Slices(sl).io.DinTop_T_7b_L(tc)(dc) := PreAdder_4b8b(sl)(tc)(dc).io.T_7b_L
        Slices(sl).io.DinTop_T_7b_R(tc)(dc) := PreAdder_4b8b(sl)(tc)(dc).io.T_7b_R
      }
    }
  }


  // * Horizontal
  for (sl <- 0 until Slice_Num) {
    for (tr <- 0 until Tile_Row) {
      for (dr <- 0 until DualPE_Row) {
        io.PSumOut_L(sl)(tr)(dr).setName(s"PSumOut_L_sl${sl}_tr${tr}_dr${dr}") := Slices(sl).io.PSumOut_L(tr)(dr)
        io.PSumOut_R(sl)(tr)(dr).setName(s"PSumOut_R_sl${sl}_tr${tr}_dr${dr}") := Slices(sl).io.PSumOut_R(tr)(dr)
      }
    }
  }

}


// Generate Verilog Codes
object UniCore_SA_GEMM_4b8b_Gen extends App {
  Config.setGenSubDir("/UniCore/UniCore_SA_GEMM_4b8b")
  Config.spinal.generateVerilog(
    // 1 Tiles = 4 rows * 2*2 cols = 4 rows * 4 cols PEs,
    // (16 * 8) Tiles = 64 rows * 32 cols PEs = 1 Slices,
    // 2 Slices = 64 rows * 64 cols PEs = 1 GEMM Unit
    UniCore_SA_GEMM_4b8b(Slice_Num=2, Tile_Row=16, Tile_Col=8, DualPE_Row=4, DualPE_Col=2, AccuWidth=18)
  ).mergeRTLSource()
}