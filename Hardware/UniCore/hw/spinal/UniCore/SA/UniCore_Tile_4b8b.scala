package UniCore.SA

import spinal.core._
import spinal.core.sim._
import UniCore.Config
import scala.language.postfixOps
import UniCore.ComposablePE.Scalable_4b8b.{PreAdd_4b8b, ScalablePE_4b8b}


// * UniCore Tile (Per-Group Quantization Supported, Systolic Spatial Array, Weight Stationary)
case class UniCore_Tile_4b8b(
                      DualPE_Row : Int,
                      DualPE_Col : Int,
                      AccuWidth  : Int
                    ) extends Component {
  val io = new Bundle {
    val W_6b_CIN_L    = in  Vec(Bits(6 bits), DualPE_Col)            // 6b = [Sign] + 5b (E & M)
    val W_6b_CIN_R    = in  Vec(Bits(6 bits), DualPE_Col)            // 6b = [Sign] + 5b (E & M)
    val W_6b_COUT_L   = out Vec(Bits(6 bits), DualPE_Col)            // 6b = [Sign] + 5b (E & M)
    val W_6b_COUT_R   = out Vec(Bits(6 bits), DualPE_Col)            // 6b = [Sign] + 5b (E & M)

    val T_7b_CIN_L    = in  Vec(Bits(7 bits), DualPE_Col)            // 7b = [Sign] + 6b (E & M)
    val T_7b_CIN_R    = in  Vec(Bits(7 bits), DualPE_Col)            // 7b = [Sign] + 6b (E & M)
    val T_7b_COUT_L   = out Vec(Bits(7 bits), DualPE_Col)            // 7b = [Sign] + 6b (E & M)
    val T_7b_COUT_R   = out Vec(Bits(7 bits), DualPE_Col)            // 7b = [Sign] + 6b (E & M)

    val PE_Mode       = in  Bool()                                   // MARK: 0 for W4A4, 1 for W8A8
    val WqLock        = in  Bool()                                   // Locking the preloaded Wq

    val PSumIn_L      = in  Vec(SInt(AccuWidth bits), DualPE_Row)    // Dual Chain for Partial Sum
    val PSumIn_R      = in  Vec(SInt(AccuWidth bits), DualPE_Row)    // Dual Chain for Partial Sum
    val PSumOut_L     = out Vec(SInt(AccuWidth bits), DualPE_Row)
    val PSumOut_R     = out Vec(SInt(AccuWidth bits), DualPE_Row)
  }
  noIoPrefix()


  // * Generate DualPEs (Each ScalablePE_4b8b can be seen as a "Dual PE")
  val DualPEs = List.tabulate(DualPE_Row,DualPE_Col)((dr, dc) => { ScalablePE_4b8b(
    AccuWidth=AccuWidth
  ).setName(s"DualPEs_dr${dr}_dc${dc}") })


  // * Vertical (T)
  for (dc <- 0 until DualPE_Col) {
    for (dr <- 0 until DualPE_Row) {
      if (dr == 0) {
        DualPEs(dr)(dc).io.T_7b_CIN_L := io.T_7b_CIN_L(dc).setName(s"T_7b_CIN_L_dc${dc}")
        DualPEs(dr)(dc).io.T_7b_CIN_R := io.T_7b_CIN_R(dc).setName(s"T_7b_CIN_R_dc${dc}")
      } else {
        DualPEs(dr)(dc).io.T_7b_CIN_L := DualPEs(dr-1)(dc).io.T_7b_COUT_L
        DualPEs(dr)(dc).io.T_7b_CIN_R := DualPEs(dr-1)(dc).io.T_7b_COUT_R
      }
    }
    io.T_7b_COUT_L(dc).setName(s"T_7b_COUT_L_dc${dc}") := DualPEs(DualPE_Row-1)(dc).io.T_7b_COUT_L
    io.T_7b_COUT_R(dc).setName(s"T_7b_COUT_R_dc${dc}") := DualPEs(DualPE_Row-1)(dc).io.T_7b_COUT_R
  }


  // * Vertical: Weight Stationary RegTable
  val WghtTable_L = List.tabulate(DualPE_Row,DualPE_Col)((dr, dc) => { Reg(Bits(6 bits)).init(B(0)).setName(s"WghtTable_dr${dr}_dc${dc}_L") })
  val WghtTable_R = List.tabulate(DualPE_Row,DualPE_Col)((dr, dc) => { Reg(Bits(6 bits)).init(B(0)).setName(s"WghtTable_dr${dr}_dc${dc}_R") })

  for (dc <- 0 until DualPE_Col) {
    for (dr <- 0 until DualPE_Row) {
      if (dr == 0) {
        when(io.WqLock) {
          WghtTable_L(dr)(dc) := WghtTable_L(dr)(dc)
          WghtTable_R(dr)(dc) := WghtTable_R(dr)(dc)
        } otherwise {
          WghtTable_L(dr)(dc) := io.W_6b_CIN_L(dc).setName(s"W_6b_CIN_dc${dc}_L")
          WghtTable_R(dr)(dc) := io.W_6b_CIN_R(dc).setName(s"W_6b_CIN_dc${dc}_R")
        }
      } else {
        when(io.WqLock) {
          WghtTable_L(dr)(dc) := WghtTable_L(dr)(dc)
          WghtTable_R(dr)(dc) := WghtTable_R(dr)(dc)
        } otherwise {
          WghtTable_L(dr)(dc) := WghtTable_L(dr-1)(dc)
          WghtTable_R(dr)(dc) := WghtTable_R(dr-1)(dc)
        }
      }
    }
    io.W_6b_COUT_L(dc).setName(s"W_6b_COUT_dc${dc}_L") := WghtTable_L(DualPE_Row-1)(dc)
    io.W_6b_COUT_R(dc).setName(s"W_6b_COUT_dc${dc}_R") := WghtTable_R(DualPE_Row-1)(dc)
  }

  // * Vertical (Wght)
  for (dr <- 0 until DualPE_Row) {
    for (dc <- 0 until DualPE_Col) {
      DualPEs(dr)(dc).io.W_6b_L := WghtTable_L(dr)(dc)
      DualPEs(dr)(dc).io.W_6b_R := WghtTable_R(dr)(dc)
    }
  }


  // * Horizontal (PSum)
  for (dr <- 0 until DualPE_Row) {
    for (dc <- 0 until DualPE_Col) {
      if (dc == 0) {
        DualPEs(dr)(dc).io.PSumIn_L := io.PSumIn_L(dc).setName(s"PSumIn_L_dr${dr}")
        DualPEs(dr)(dc).io.PSumIn_R := io.PSumIn_R(dc).setName(s"PSumIn_R_dr${dr}")
      } else {
        DualPEs(dr)(dc).io.PSumIn_L := DualPEs(dr)(dc-1).io.PSumOut_L
        DualPEs(dr)(dc).io.PSumIn_R := DualPEs(dr)(dc-1).io.PSumOut_R
      }
    }
    io.PSumOut_L(dr).setName(s"PSumOut_L_dr${dr}") := DualPEs(dr)(DualPE_Col-1).io.PSumOut_L
    io.PSumOut_R(dr).setName(s"PSumOut_R_dr${dr}") := DualPEs(dr)(DualPE_Col-1).io.PSumOut_R
  }


  // * One to all
  DualPEs.foreach{ pr => pr.foreach{ pc =>
    pc.io.PE_Mode := io.PE_Mode
  }}

}