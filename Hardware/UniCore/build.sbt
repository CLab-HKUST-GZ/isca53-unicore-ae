ThisBuild / version := "1.0"
ThisBuild / scalaVersion := "2.13.14"
ThisBuild / organization := "org.example"

val spinalVersion = "1.12.3"
val spinalCore = "com.github.spinalhdl" %% "spinalhdl-core" % spinalVersion
val spinalLib = "com.github.spinalhdl" %% "spinalhdl-lib" % spinalVersion
val spinalIdslPlugin = compilerPlugin("com.github.spinalhdl" %% "spinalhdl-idsl-plugin" % spinalVersion)

// lazy val projectname = (project in file("."))
//   .settings(
//     name := "myproject", 
//     Compile / scalaSource := baseDirectory.value / "hw" / "spinal",
//     libraryDependencies ++= Seq(spinalCore, spinalLib, spinalIdslPlugin)
//   )

lazy val projectname = (project in file("."))
  .settings(
    Compile / scalaSource := baseDirectory.value / "hw" / "spinal",
    libraryDependencies ++= Seq(
      spinalCore,
      spinalLib,
      spinalIdslPlugin,
      "org.playframework" %% "play-json" % "3.0.0"    // or try "2.9.4"
    )
  )


fork := true
