package UniCore.Testing.GoldenModels
import UniCore.Convert.{Bin2FPCvt, FP2BinCvt}
import scala.math.pow


object SFPMA_8b {
  def calculate(
                 W_FP8_In : BigInt,
                 A_FP8_In : BigInt,
               ): String = {

    val w_int = W_FP8_In.toInt
    val a_int = A_FP8_In.toInt

    // 1. Unpack inputs: Extract sign bit and exponent-mantissa part
    val sign_w = (w_int >> 7) & 0x1     // bit [7]
    val sign_a = (a_int >> 7) & 0x1     // bit [7]
    val expo_mant_w = w_int & 0x7F      // bits [6:0]
    val expo_mant_a = a_int & 0x7F      // bits [6:0]

    // 2. Extract table lookup indices: bits [2:0] (E4M3 mantissa is 3 bits)
    val idx_w = w_int & 0x7
    val idx_a = a_int & 0x7

    // 3. Look up both Down-Sampling (DS) and Up-Scaling (US) compensation values
    val comp_ds = CompTable.M3_DS(idx_a, idx_w)   // Fed into Adder Cin
    val comp_us = CompTable.M3_US(idx_a, idx_w)   // Appended as LSB

    // 4. Calculate Correction (-Bias)
    val minus_b = -(pow(2, 4-1)-1).toInt  // -Bias (FP8_E4M3) = -7

    // Shift left by 3 bits (mantissa width) and mask to 8 bits simulating HW truncation
    val minus_b_shifted = (minus_b << 3) & 0xFF

    // 5. Execute Hardware Stage 1 & 2 Additions
    val sum_8b = (expo_mant_a + expo_mant_w + minus_b_shifted + comp_ds) & 0xFF

    // 6. Construct the 9-bit Hyper-Resolution Exponent-Mantissa (E5M4)
    val r_expo_mant_9b = (sum_8b << 1) | comp_us

    // 7. Calculate Sign Bit
    val r_sign = sign_a ^ sign_w

    // 8. Combine into the final 10-bit S1E5M4 result
    val r_int = (r_sign << 9) | r_expo_mant_9b

    // 9. Format the result as a 10-bit binary string (zero-padded at the MSB)
    val R_FPBin = String.format("%10s", r_int.toBinaryString).replace(' ', '0')

    R_FPBin
  }
}