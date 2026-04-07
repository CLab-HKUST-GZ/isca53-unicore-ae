package UniCore.Compensation

import spinal.core._
import spinal.core.sim._
import UniCore.Config
import scala.language.postfixOps


case class Comp_M2_US() extends Component {
  val io = new Bundle {
    val A_M2     = in  Bits(2 bits)
    val W_M2     = in  Bits(2 bits)
    val HyperRes = out Bits(2 bits)
  }
  noIoPrefix()

  val InIdx = io.A_M2 ## io.W_M2

  switch(InIdx) {
    is(B"10_01") { io.HyperRes := B"10" }
    is(B"10_10") { io.HyperRes := B"10" }
    is(B"10_11") { io.HyperRes := B"01" }
    default      { io.HyperRes := B"00" }
  }

}


object Comp_M2_US_Gen extends App {
  Config.setGenSubDir("/UniCore")
  Config.spinal.generateVerilog(Comp_M2_US()).printRtl().mergeRTLSource()
}