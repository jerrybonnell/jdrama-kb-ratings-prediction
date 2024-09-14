import java.util.*;
import java.io.*;
/**
 * comparing file objects based on their names
 */
public class FileComp implements Comparator<File> {
  /* ********************************************
   * compare two objects a and b using compareTo()
   * concerning their names
   * ******************************************** */
  @Override
  public int compare( File a, File b ) {
    return a.getName().compareTo( b.getName() );
  }
}
