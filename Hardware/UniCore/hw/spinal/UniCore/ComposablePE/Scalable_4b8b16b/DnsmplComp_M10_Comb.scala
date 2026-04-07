package UniCore.ComposablePE.Scalable_4b8b16b

import spinal.core._
import spinal.core.sim._
import UniCore.Config
import scala.language.postfixOps


case class DnsmplComp_M10_Comb() extends Component {
  val io = new Bundle {
    val A_idx = in  Bits(3 bits)
    val W_idx = in  Bits(3 bits)
    val Comp  = out Bits(8 bits)
  }
  noIoPrefix()

  val Idx = io.A_idx ## io.W_idx

  switch(Idx) {
    is(B"000_000") { io.Comp := B(4  , 8 bits) }
    is(B"000_001") { io.Comp := B(12 , 8 bits) }
    is(B"000_010") { io.Comp := B(20 , 8 bits) }
    is(B"000_011") { io.Comp := B(28 , 8 bits) }
    is(B"000_100") { io.Comp := B(36 , 8 bits) }
    is(B"000_101") { io.Comp := B(44 , 8 bits) }
    is(B"000_110") { io.Comp := B(48 , 8 bits) }
    is(B"000_111") { io.Comp := B(24 , 8 bits) }

    is(B"001_000") { io.Comp := B(12 , 8 bits) }
    is(B"001_001") { io.Comp := B(36 , 8 bits) }
    is(B"001_010") { io.Comp := B(60 , 8 bits) }
    is(B"001_011") { io.Comp := B(84 , 8 bits) }
    is(B"001_100") { io.Comp := B(108, 8 bits) }
    is(B"001_101") { io.Comp := B(115, 8 bits) }
    is(B"001_110") { io.Comp := B(78 , 8 bits) }
    is(B"001_111") { io.Comp := B(26 , 8 bits) }

    is(B"010_000") { io.Comp := B(20 , 8 bits) }
    is(B"010_001") { io.Comp := B(60 , 8 bits) }
    is(B"010_010") { io.Comp := B(100, 8 bits) }
    is(B"010_011") { io.Comp := B(139, 8 bits) }
    is(B"010_100") { io.Comp := B(148, 8 bits) }
    is(B"010_101") { io.Comp := B(110, 8 bits) }
    is(B"010_110") { io.Comp := B(66 , 8 bits) }
    is(B"010_111") { io.Comp := B(22 , 8 bits) }

    is(B"011_000") { io.Comp := B(28 , 8 bits) }
    is(B"011_001") { io.Comp := B(84 , 8 bits) }
    is(B"011_010") { io.Comp := B(139, 8 bits) }
    is(B"011_011") { io.Comp := B(158, 8 bits) }
    is(B"011_100") { io.Comp := B(126, 8 bits) }
    is(B"011_101") { io.Comp := B(90 , 8 bits) }
    is(B"011_110") { io.Comp := B(54 , 8 bits) }
    is(B"011_111") { io.Comp := B(18 , 8 bits) }

    is(B"100_000") { io.Comp := B(36 , 8 bits) }
    is(B"100_001") { io.Comp := B(108, 8 bits) }
    is(B"100_010") { io.Comp := B(148, 8 bits) }
    is(B"100_011") { io.Comp := B(126, 8 bits) }
    is(B"100_100") { io.Comp := B(98 , 8 bits) }
    is(B"100_101") { io.Comp := B(70 , 8 bits) }
    is(B"100_110") { io.Comp := B(42 , 8 bits) }
    is(B"100_111") { io.Comp := B(14 , 8 bits) }

    is(B"101_000") { io.Comp := B(44 , 8 bits) }
    is(B"101_001") { io.Comp := B(115, 8 bits) }
    is(B"101_010") { io.Comp := B(110, 8 bits) }
    is(B"101_011") { io.Comp := B(90 , 8 bits) }
    is(B"101_100") { io.Comp := B(70 , 8 bits) }
    is(B"101_101") { io.Comp := B(50 , 8 bits) }
    is(B"101_110") { io.Comp := B(30 , 8 bits) }
    is(B"101_111") { io.Comp := B(10 , 8 bits) }

    is(B"110_000") { io.Comp := B(48 , 8 bits) }
    is(B"110_001") { io.Comp := B(78 , 8 bits) }
    is(B"110_010") { io.Comp := B(66 , 8 bits) }
    is(B"110_011") { io.Comp := B(54 , 8 bits) }
    is(B"110_100") { io.Comp := B(42 , 8 bits) }
    is(B"110_101") { io.Comp := B(30 , 8 bits) }
    is(B"110_110") { io.Comp := B(18 , 8 bits) }
    is(B"110_111") { io.Comp := B(6  , 8 bits) }

    is(B"111_000") { io.Comp := B(24 , 8 bits) }
    is(B"111_001") { io.Comp := B(26 , 8 bits) }
    is(B"111_010") { io.Comp := B(22 , 8 bits) }
    is(B"111_011") { io.Comp := B(18 , 8 bits) }
    is(B"111_100") { io.Comp := B(14 , 8 bits) }
    is(B"111_101") { io.Comp := B(10 , 8 bits) }
    is(B"111_110") { io.Comp := B(6  , 8 bits) }
    is(B"111_111") { io.Comp := B(2  , 8 bits) }
  }

}


// COMP_TABLE_8x8 = np.array([
//   [  4,  12,  20,  28,  36,  44,  48,  24],
//   [ 12,  36,  60,  84, 108, 115,  78,  26],
//   [ 20,  60, 100, 139, 148, 110,  66,  22],
//   [ 28,  84, 139, 158, 126,  90,  54,  18],
//   [ 36, 108, 148, 126,  98,  70,  42,  14],
//   [ 44, 115, 110,  90,  70,  50,  30,  10],
//   [ 48,  78,  66,  54,  42,  30,  18,   6],
//   [ 24,  26,  22,  18,  14,  10,   6,   2],
// ], dtype=np.int32)



object DnsmplComp_M10_Comb_Gen extends App {
  Config.setGenSubDir("/UniCore")
  Config.spinal.generateVerilog(DnsmplComp_M10_Comb()).printRtl().mergeRTLSource()
}


