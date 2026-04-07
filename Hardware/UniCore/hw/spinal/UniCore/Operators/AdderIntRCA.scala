package UniCore.Operators

import spinal.core._
import spinal.core.sim._
import UniCore.Config
import scala.language.postfixOps


// MARK: BlackBox for add_int_rca.v file
case class add_int_rca(Width: Int) extends BlackBox {

  addGeneric("WIDTH", Width)

  val io = new Bundle {
    val Operand_1 = in  Bits(Width bits)
    val Operand_2 = in  Bits(Width bits)
    val Cin       = in  Bits(1 bits)
    val Sum       = out Bits(Width bits)
    val Cout      = out Bits(1 bits)
  }
  noIoPrefix()

  // ? Be careful to the blackbox import path
  addRTLPath(s"hw/spinal/UniCore/BlackBoxImport/add_int_rca.v")

}



case class AdderIntRCA(Width: Int) extends Component {
  val io = new Bundle {
    val Operand_1 = in  Bits(Width bits)
    val Operand_2 = in  Bits(Width bits)
    val Cin       = in  Bits(1 bits)
    val Sum       = out Bits(Width bits)
    val Cout      = out Bits(1 bits)
  }
  noIoPrefix()

  val Adder = new add_int_rca(Width)
  Adder.io.Operand_1 := io.Operand_1
  Adder.io.Operand_2 := io.Operand_2
  Adder.io.Cin := io.Cin
  io.Sum := Adder.io.Sum
  io.Cout := Adder.io.Cout

}



object AdderIntRCA_Gen extends App {
  Config.setGenSubDir("/UniCore")
  Config.spinal.generateVerilog(AdderIntRCA(Width=10)).printRtl().mergeRTLSource()
}