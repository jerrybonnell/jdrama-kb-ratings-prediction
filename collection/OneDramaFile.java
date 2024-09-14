import java.util.*;
import java.io.*;
/**
 * class for extracting DrameObjects from a single file
 */
public class OneDramaFile {
  ArrayList<DramaObject> objectList;
  File inFile;
  String season;

  /**
   * constructor
   */
  OneDramaFile() {
    objectList = new ArrayList<DramaObject>();
  }

  /**
   * the method for reading DramaObject data
   * @param	f	the input File
   * @param	s	the season name
   */
  public void read( File f, String s ) throws FileNotFoundException {
    //// save the two parameters and initialize the list
    inFile = f;
    season = s;
    LinkedList<String> lines = new LinkedList<String>();

    //// get reader to read from the file
    Scanner in = new Scanner( f );
    //// the variable stating if reading is in a DramaObject section
    boolean inside = false;

    String w;
    DramaObject object;

    while ( in.hasNextLine() ) {
      w = in.nextLine();
      /* ************************************************************
       * if inside, add the line to the linked list of String data
       *   if the line contains "</table>", the block ends,
       *   place it in DramaObject, and add the result to the object
       *   list, and turn inside off
       * otherwise, look for "<a name=" as the starting point
       ************************************************************ */
      if ( inside ) {
        lines.add( w );
        if ( w.indexOf( "</table>" ) >= 0 ) {
          inside = false;
          object = new DramaObject( s );
          object.analyze( lines );
          objectList.add( object );
        }
      }
      else {
        if ( w.indexOf( "<a name=" ) >= 0 ) {
          inside = true;
          lines = new LinkedList<String>();
          lines.add( w );
        }
      }
    }
    in.close();
  }

  /**
   * print all information
   */
  public void print() {
    for ( DramaObject obj : objectList ) {
      obj.print();
    }
  }

  /**
   * encode all as entries in a csv file
   * @param	st	the PrintStream to which to print
   */
  public void encode( PrintStream st ) {
    for ( DramaObject obj : objectList ) {
      st.println(  obj.encode() );
    }
  }
}
