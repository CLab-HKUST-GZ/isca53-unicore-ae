package UniCore.SA

import spinal.core._
import spinal.core.sim._
import UniCore.Config
import scala.language.postfixOps


// * UniCore Slice (Per-Group Quantization Supported, Systolic Spatial Array, Weight Stationary)
case class UniCore_Slice_4b8b(
                               Tile_Row   : Int,
                               Tile_Col   : Int,
                               DualPE_Row : Int,
                               DualPE_Col : Int,
                               AccuWidth  : Int
                             ) extends Component {
  val io = new Bundle {
    val DinTop_W_6b_L = in  Vec(Vec(Bits(6 bits), DualPE_Col), Tile_Col)            // 6b = [Sign] + 5b (E & M)
    val DinTop_W_6b_R = in  Vec(Vec(Bits(6 bits), DualPE_Col), Tile_Col)            // 6b = [Sign] + 5b (E & M)

    val DinTop_T_7b_L = in  Vec(Vec(Bits(7 bits), DualPE_Col), Tile_Col)            // 7b = [Sign] + 6b (E & M)
    val DinTop_T_7b_R = in  Vec(Vec(Bits(7 bits), DualPE_Col), Tile_Col)            // 7b = [Sign] + 6b (E & M)

    val PE_Mode       = in  Bool()                                                  // MARK: 0 for 3/4bits-PE, 1 for 6/8bits-PE
    val WqLock        = in  Bool()                                                  // Locking the preloaded Wq

    val PSumOut_L     = out Vec(Vec(SInt(AccuWidth bits), DualPE_Row), Tile_Row)
    val PSumOut_R     = out Vec(Vec(SInt(AccuWidth bits), DualPE_Row), Tile_Row)
  }
  noIoPrefix()


  // * Generate Tiles
  val Tiles = List.tabulate(Tile_Row,Tile_Col)((tr, tc) => { UniCore_Tile_4b8b(
    DualPE_Row=DualPE_Row, DualPE_Col=DualPE_Col, AccuWidth=AccuWidth
  ).setName(s"Tiles_tr${tr}_tc${tc}") })


  // * Generate tables of size TileRow*TileCol for DinTop Regs each is Vec(Regs, DualPE_Col)
  val DinTopRegTable_T_7b_L = List.tabulate(Tile_Row,Tile_Col)((tr, tc) => {
    Vec((0 until DualPE_Col).map{ dc => Reg(Bits(7 bits)).init(B(0)).setName(s"DinTopRegTable_T_7b_L_tr${tr}_tc${tc}_dc${dc}") })
  })    // for pipelining

  val DinTopRegTable_T_7b_R = List.tabulate(Tile_Row,Tile_Col)((tr, tc) => {
    Vec((0 until DualPE_Col).map{ dc => Reg(Bits(7 bits)).init(B(0)).setName(s"DinTopRegTable_T_7b_R_tr${tr}_tc${tc}_dc${dc}") })
  })    // for pipelining


  // * Generate a table of size (TileRow-1)*TileCol for Horizontal PSum Regs passing, each is Vec(Regs, DualPE_Row)
  val HorizRegTable_PSum_L = List.tabulate(Tile_Row,Tile_Col-1)((tr, tc) => {
    Vec((0 until DualPE_Row).map{ dr => Reg(SInt(AccuWidth bits)).init(S(0)).setName(s"HorizRegTable_PSum_L_tr${tr}_tc${tc}_dr${dr}") })
  })    // for pipelining

  val HorizRegTable_PSum_R = List.tabulate(Tile_Row,Tile_Col-1)((tr, tc) => {
    Vec((0 until DualPE_Row).map{ dr => Reg(SInt(AccuWidth bits)).init(S(0)).setName(s"HorizRegTable_PSum_R_tr${tr}_tc${tc}_dr${dr}") })
  })    // for pipelining


  // * Vertical
  for (tc <- 0 until Tile_Col) {
    Tiles(0)(tc).io.W_6b_CIN_L := io.DinTop_W_6b_L(tc)
    Tiles(0)(tc).io.W_6b_CIN_R := io.DinTop_W_6b_R(tc)
    Tiles(0)(tc).io.T_7b_CIN_L := io.DinTop_T_7b_L(tc)
    Tiles(0)(tc).io.T_7b_CIN_R := io.DinTop_T_7b_R(tc)
    for (tr <- 0 until Tile_Row-1) {
      Tiles(tr+1)(tc).io.W_6b_CIN_L := Tiles(tr)(tc).io.W_6b_COUT_L
      Tiles(tr+1)(tc).io.W_6b_CIN_R := Tiles(tr)(tc).io.W_6b_COUT_R
      Tiles(tr+1)(tc).io.T_7b_CIN_L := Tiles(tr)(tc).io.T_7b_COUT_L
      Tiles(tr+1)(tc).io.T_7b_CIN_R := Tiles(tr)(tc).io.T_7b_COUT_R
    }
  }


  // * Horizontal
  for (tr <- 0 until Tile_Row) {
    Tiles(tr)(0).io.PSumIn_L := Vec(S(0), DualPE_Row)
    Tiles(tr)(0).io.PSumIn_R := Vec(S(0), DualPE_Row)
    for (tc <- 0 until Tile_Col-1) {
      HorizRegTable_PSum_L(tr)(tc) := Tiles(tr)(tc).io.PSumOut_L
      HorizRegTable_PSum_R(tr)(tc) := Tiles(tr)(tc).io.PSumOut_R
      Tiles(tr)(tc+1).io.PSumIn_L := HorizRegTable_PSum_L(tr)(tc)
      Tiles(tr)(tc+1).io.PSumIn_R := HorizRegTable_PSum_R(tr)(tc)
    }
    // Outputs
    io.PSumOut_L(tr) := Tiles(tr)(Tile_Col-1).io.PSumIn_L
    io.PSumOut_R(tr) := Tiles(tr)(Tile_Col-1).io.PSumIn_R
  }


  // * One to all
  Tiles.foreach{ tr => tr.foreach{ tc =>
    tc.io.PE_Mode := io.PE_Mode
    tc.io.WqLock := io.WqLock
  }}


  // * Better naming
  (0 until Tile_Col).foreach{tc => (0 until DualPE_Col).foreach{dc =>
    io.DinTop_W_6b_L(tc)(dc).setName(s"DinTop_W_6b_L_tc${tc}_dc${dc}")
    io.DinTop_W_6b_R(tc)(dc).setName(s"DinTop_W_6b_R_tc${tc}_dc${dc}")
    io.DinTop_T_7b_L(tc)(dc).setName(s"DinTop_T_7b_L_tc${tc}_dc${dc}")
    io.DinTop_T_7b_R(tc)(dc).setName(s"DinTop_T_7b_R_tc${tc}_dc${dc}")
  }}

}