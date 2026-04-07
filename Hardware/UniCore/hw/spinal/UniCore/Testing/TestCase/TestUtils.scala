package UniCore.Testing.TestCase
import UniCore.Convert.{Bin2FPCvt, FP2BinCvt}
import scala.math.pow

// Test Utils
object TU {

  // * A4 in FP4 (E2M1), converted into S1E3M2 form
  val A4_ValueSpaceAsE3M2 = List(
    // Positive
    "0_010_00",    // Value =  0.5 , ("0_00_1" in FP4)
    "0_011_00",    // Value =  1.0 , ("0_01_0" in FP4)
    "0_011_10",    // Value =  1.5 , ("0_01_1" in FP4)
    "0_100_00",    // Value =  2.0 , ("0_10_0" in FP4)
    "0_100_10",    // Value =  3.0 , ("0_10_1" in FP4)
    "0_101_00",    // Value =  4.0 , ("0_11_0" in FP4)
    "0_101_10",    // Value =  6.0 , ("0_11_1" in FP4)
    // Negative
    "1_010_00",    // Value = -0.5 , ("1_00_1" in FP4)
    "1_011_00",    // Value = -1.0 , ("1_01_0" in FP4)
    "1_011_10",    // Value = -1.5 , ("1_01_1" in FP4)
    "1_100_00",    // Value = -2.0 , ("1_10_0" in FP4)
    "1_100_10",    // Value = -3.0 , ("1_10_1" in FP4)
    "1_101_00",    // Value = -4.0 , ("1_11_0" in FP4)
    "1_101_10",    // Value = -6.0 , ("1_11_1" in FP4)
  )

  // * A4 in FP4 (E2M1), converted into Integer form for simulation input
  val A4_ValueSpace = A4_ValueSpaceAsE3M2.map(_.replace("_", "")).map(Integer.parseInt(_, 2))

  // * W in DynFP3 or DynFP4, merged value space and all converted into S1E3M2 form
  val W4_ValueSpaceAsE3M2 = List(
                   //       |     FP3 Variants     |            FP4 Variants             |               With I-flag              | Z-Value |
                   // ------|----------------------|-------------------------------------|----------------------------------------|---------|
                   // Value | FP3(E2M0)| FP3(E1M1) | FP4(E3M0) |  FP4(E2M1) |  FP4(E1M2) |  FP3(E2M0I) | FP3(E1M1I) |  FP4(E1M2I) | Support |
                   // ------|----------------------|-------------------------------------|----------------------------------------|---------|
    "0_001_00",    // 0.25  |          |           |  "0_001"  |            |            |             |            |             |    N    |
    "0_010_00",    // 0.5   |          |           |  "0_010"  |  "0_00_1"  |  "0_0_01"  |             |            | "0_0(0)_01" |    Y    |
    "0_010_01",    // 0.625 |          |           |           |            |            |             |            |             |    Y    |
    "0_010_10",    // 0.75  |          |           |           |            |            |             |            |             |    Y    |
    "0_010_11",    // 0.875 |          |           |           |            |            |             |            |             |    Y    |
    "0_011_00",    // 1.0   |          |  "0_0_1"  |  "0_011"  |  "0_01_0"  |  "0_0_10"  |  "0_0(0)1"  | "0_0(0)_1" | "0_0(0)_10" |    Y    |
    "0_011_01",    // 1.25  |          |           |           |            |            |             |            |             |    Y    |
    "0_011_10",    // 1.5   |          |           |           |  "0_01_1"  |  "0_0_11"  |             |            | "0_0(0)_11" |    Y    |
    "0_011_11",    // 1.75  |          |           |           |            |            |             |            |             |    Y    |
    "0_100_00",    // 2.0   |  "0_01"  |  "0_1_0"  |  "0_100"  |  "0_10_0"  |  "0_1_00"  |             |            |             |    Y    |
    "0_100_01",    // 2.5   |          |           |           |            |  "0_1_01"  |             |            |             |    Y    |
    "0_100_10",    // 3.0   |          |  "0_1_1"  |           |  "0_10_1"  |  "0_1_10"  |             |            |             |    Y    |
    "0_100_11",    // 3.5   |          |           |           |            |  "0_1_11"  |             |            |             |    Y    |
    "0_101_00",    // 4.0   |  "0_10"  |           |  "0_101"  |  "0_11_0"  |            |             | "0_1(0)_0" | "0_1(0)_00" |    Y    |
    "0_101_01",    // 5.0   |          |           |           |            |            |             |            | "0_1(0)_01" |    Y    |
    "0_101_10",    // 6.0   |          |           |           |  "0_11_1"  |            |             | "0_1(0)_1" | "0_1(0)_10" |    Y    |
    "0_101_11",    // 7.0   |          |           |           |            |            |             |            | "0_1(0)_11" |    Y    |
    "0_110_00",    // 8.0   |  "0_11"  |           |  "0_110"  |            |            |  "0_1(0)0"  |            |             |    Y    |
    "0_110_01",    // 10.0  |          |           |           |            |            |             |            |             |    Y    |
    "0_110_10",    // 12.0  |          |           |           |            |            |             |            |             |    Y    |
    "0_110_11",    // 14.0  |          |           |           |            |            |             |            |             |    Y    |
    "0_111_00",    // 16.0  |          |           |  "0_111"  |            |            |  "0_1(0)1"  |            |             |    Y    |
    "0_111_01",    // 20.0  |          |           |           |            |            |             |            |             |    Y    |
    "0_111_10",    // 24.0  |          |           |           |            |            |             |            |             |    Y    |
    "0_111_11",    // 28.0  |          |           |           |            |            |             |            |             |    Y    |
  )

  // * W in DynFP3 or DynFP4, merged value space and all converted into Integer form for simulation input
  val W4_ValueSpace = W4_ValueSpaceAsE3M2.map(_.replace("_", "")).map(Integer.parseInt(_, 2))

  // * Result Checking and Logging
  def checkResult(isCorrect: Boolean): String = {
    if (isCorrect) {
      "\u001B[32mCorrect\u001B[0m"
    } else {
      "\u001B[31mIncorrect\u001B[0m"
    }
  }

}