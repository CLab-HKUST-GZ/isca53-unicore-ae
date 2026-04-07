package UniCore.Testing.GoldenModels
import UniCore.Convert.{Bin2FPCvt, FP2BinCvt}


object CompTable {

  // * M=2, Up-Scaling Compensation
  def M2_US(A_M2: String, W_M2: String): String = {
    (A_M2, W_M2) match {
      case ("10", "01") => "10"
      case ("10", "10") => "10"
      case ("10", "11") => "01"
      case _            => "00"
    }
  }

  // * M=3, Up-Scaling Compensation
  def M3_US(A_M3: Int, W_M3: Int): Int = {
    val lookupTable: Vector[Vector[Int]] = Vector(
      Vector(0, 0, 0, 0, 0, 0, 0, 0),
      Vector(0, 0, 0, 1, 1, 1, 1, 1),
      Vector(0, 0, 0, 0, 0, 0, 1, 1),
      Vector(0, 1, 0, 0, 1, 0, 1, 1),
      Vector(0, 1, 0, 1, 0, 0, 1, 0),
      Vector(0, 1, 0, 0, 0, 1, 1, 0),
      Vector(0, 1, 1, 1, 1, 1, 0, 0),
      Vector(0, 1, 1, 1, 0, 0, 0, 0),
    )

    require(A_M3 >= 0 && A_M3 <= 7, "Index out of range")
    require(W_M3 >= 0 && W_M3 <= 7, "Index out of range")

    lookupTable(A_M3)(W_M3)
  }

  // * M=3, Down-Sampling Compensation
  def M3_DS(A_M3: Int, W_M3: Int): Int = {
    val lookupTable: Vector[Vector[Int]] = Vector(
      Vector(0, 0, 0, 0, 0, 0, 0, 0),
      Vector(0, 0, 0, 0, 0, 0, 0, 0),
      Vector(0, 0, 0, 1, 1, 1, 0, 0),
      Vector(0, 0, 1, 1, 1, 1, 0, 0),
      Vector(0, 0, 1, 1, 1, 1, 0, 0),
      Vector(0, 0, 1, 1, 1, 0, 0, 0),
      Vector(0, 0, 0, 0, 0, 0, 0, 0),
      Vector(0, 0, 0, 0, 0, 0, 0, 0),
    )

    require(A_M3 >= 0 && A_M3 <= 7, "A_M3 out of range")
    require(W_M3 >= 0 && W_M3 <= 7, "W_M3 out of range")

    lookupTable(A_M3)(W_M3)
  }

  // * M=10, Down-Sampling Compensation
  def M10_DS(A_M10_MSBs: Int, W_M10_MSBs: Int): Int = {
    val lookupTable: Vector[Vector[Int]] = Vector(
      Vector( 4,  12,  20,  28,  36,  44,  48,  24),
      Vector(12,  36,  60,  84, 108, 115,  78,  26),
      Vector(20,  60, 100, 139, 148, 110,  66,  22),
      Vector(28,  84, 139, 158, 126,  90,  54,  18),
      Vector(36, 108, 148, 126,  98,  70,  42,  14),
      Vector(44, 115, 110,  90,  70,  50,  30,  10),
      Vector(48,  78,  66,  54,  42,  30,  18,   6),
      Vector(24,  26,  22,  18,  14,  10,   6,   2),
    )

    require(A_M10_MSBs >= 0 && A_M10_MSBs <= 7, "A_M10_MSBs out of range")
    require(W_M10_MSBs >= 0 && W_M10_MSBs <= 7, "W_M10_MSBs out of range")

    lookupTable(A_M10_MSBs)(W_M10_MSBs)
  }


}
