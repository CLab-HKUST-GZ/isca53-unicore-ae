package UniCore.Testing.GoldenModels
import UniCore.Convert.{Bin2FPCvt, FP2BinCvt}
import scala.math.pow


object SFPMA_16b {
  def calculate(
                 W_FP16_In : BigInt,
                 A_FP16_In : BigInt,
               ): String = {

    val w_int = W_FP16_In.toInt
    val a_int = A_FP16_In.toInt

    // 1. Unpack inputs: Extract sign bit and exponent-mantissa part
    val sign_w = (w_int >> 15) & 0x1    // bits [15]
    val sign_a = (a_int >> 15) & 0x1    // bits [15]
    val expo_mant_w = w_int & 0x7FFF    // bits [14:0]
    val expo_mant_a = a_int & 0x7FFF    // bits [14:0]

    // 2. Extract table lookup indices: bits [9:7]
    val idx_w = (w_int >> 7) & 0x7
    val idx_a = (a_int >> 7) & 0x7

    // 3. Look up the table to get the 8-bit compensation value
    val comp = CompTable.M10_DS(idx_a, idx_w)

    // 4. Calculate Correction
    val minus_b = -(pow(2, 5-1)-1).toInt  // -Bias (FP16_E5M10) = -15

    // Extract the lower 6 bits of minus_b (i.e., 110001), shift left by 10 bits, and combine with comp
    val minus_b_6bits = minus_b & 0x3F
    val correction = (minus_b_6bits << 10) | comp

    // 5. Execute two-stage addition
    val stage1_sum = expo_mant_a + expo_mant_w
    // & 0xFFFF simulates the 16-bit hardware adder's width limit, discarding carry-out
    val stage2_sum = (stage1_sum + correction) & 0xFFFF

    // 6. Extract the result's exponent-mantissa and calculate the sign bit
    val r_expo_mant = stage2_sum & 0x7FFF
    val r_sign = sign_a ^ sign_w

    // 7. Combine into the final 16-bit result (stored as Int)
    val r_int = (r_sign << 15) | r_expo_mant

    // 8. Format the result as a 16-bit binary string (zero-padded at the MSB)
    val R_FPBin = String.format("%16s", r_int.toBinaryString).replace(' ', '0')

    R_FPBin
  }
}
